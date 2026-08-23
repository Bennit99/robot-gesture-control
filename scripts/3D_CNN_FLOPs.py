"""Deformable ResNeXt3D-101 profiling script.

This module implements a PyTorch version of the architecture described in the
prompt and prints parameter count plus FLOPs for a dummy input of shape
``(1, 3, 32, 112, 112)``.

The conv5 middle block uses a deformable 3D wrapper by default. Because the
project already ships with ``fvcore``, FLOPs are reported with
``FlopCountAnalysis`` when available.
"""

from __future__ import annotations

import argparse

import torch
from torch import nn
from torch.nn import functional as F


def _triple(value: int | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value, value)


class ConvBNReLU3d(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=_triple(kernel_size),
                stride=_triple(stride),
                padding=_triple(padding),
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )


class DeformableWarpConv3d(nn.Module):
    """A simple spatiotemporal deformable 3D convolution wrapper.

    The module predicts a dense 3D offset field with a 1x1x1 convolution, warps
    the input volume with ``grid_sample``, then applies a grouped 3x3x3
    convolution. This keeps the implementation fully PyTorch-native while
    matching the requested deformable conv5 behavior closely enough for
    training/profiling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int | tuple[int, int, int] = 1,
        groups: int = 32,
        max_offset_voxels: float = 1.0,
        use_deformable: bool = True,
    ) -> None:
        super().__init__()
        self.stride = _triple(stride)
        self.max_offset_voxels = float(max_offset_voxels)
        self.use_deformable = bool(use_deformable)

        # Keep the offset predictor as 1x1x1 so it does not violate the grouped
        # 3x3x3 convolution rule from the prompt.
        self.offset = nn.Conv3d(in_channels, 3, kernel_size=1, stride=self.stride)
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    @staticmethod
    def _identity_grid(batch_size: int, out_shape: tuple[int, int, int], device, dtype):
        theta = (
            torch.eye(3, 4, device=device, dtype=dtype)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )
        return F.affine_grid(
            theta, size=(batch_size, 1, *out_shape), align_corners=False
        )

    def _warp(self, x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        # offsets: [B, 3, D, H, W] in (dx, dy, dz) voxel units
        batch_size, _, depth, height, width = offsets.shape
        base_grid = self._identity_grid(
            batch_size, (depth, height, width), x.device, x.dtype
        )

        dx = torch.tanh(offsets[:, 0]) * self.max_offset_voxels
        dy = torch.tanh(offsets[:, 1]) * self.max_offset_voxels
        dz = torch.tanh(offsets[:, 2]) * self.max_offset_voxels

        def _norm(component: torch.Tensor, size: int) -> torch.Tensor:
            denom = max(size - 1, 1)
            return component * (2.0 / float(denom))

        # grid_sample expects (x, y, z) => (width, height, depth)
        offset_grid = torch.stack(
            (
                _norm(dx, x.shape[-1]),
                _norm(dy, x.shape[-2]),
                _norm(dz, x.shape[-3]),
            ),
            dim=-1,
        )
        grid = base_grid + offset_grid
        return F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_deformable:
            out = self.conv(x)
            out = self.bn(out)
            return self.relu(out)

        offsets = self.offset(x)
        warped = self._warp(x, offsets)
        out = self.conv(warped)
        out = self.bn(out)
        return self.relu(out)


class Bottleneck3D(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        bottleneck_channels: int,
        out_channels: int,
        stride: int | tuple[int, int, int] = 1,
        groups: int = 32,
        conv2_out_channels: int | None = None,
        use_deformable_middle: bool = False,
        deformable_middle: bool = True,
    ) -> None:
        super().__init__()
        stride = _triple(stride)
        conv2_out_channels = conv2_out_channels or bottleneck_channels

        self.conv1 = ConvBNReLU3d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
        )

        if use_deformable_middle:
            self.conv2 = DeformableWarpConv3d(
                bottleneck_channels,
                conv2_out_channels,
                stride=1,
                groups=groups,
                use_deformable=deformable_middle,
            )
        else:
            self.conv2 = ConvBNReLU3d(
                bottleneck_channels,
                conv2_out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=groups,
            )

        self.conv3 = nn.Conv3d(
            conv2_out_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn3 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if any(s != 1 for s in stride) or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        return self.relu(out)


class ResNeXt3D101(nn.Module):
    def __init__(
        self,
        num_classes: int = 27,
        dropout: float = 0.5,
        groups: int = 32,
        conv5_middle_channels: int = 2048,
        use_deformable_conv5: bool = True,
        return_softmax: bool = False,
    ) -> None:
        super().__init__()
        if conv5_middle_channels not in {1024, 2048}:
            raise ValueError("conv5_middle_channels must be 1024 or 2048")

        self.return_softmax = bool(return_softmax)
        self.stem = nn.Sequential(
            nn.Conv3d(
                3,
                64,
                kernel_size=7,
                stride=(1, 2, 2),
                padding=3,
                bias=False,
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=(1, 2, 2), padding=1),
        )

        self.layer2 = self._make_stage(
            in_channels=64,
            bottleneck_channels=128,
            conv2_out_channels=128,
            out_channels=256,
            blocks=3,
            stride=1,
            groups=groups,
            use_deformable_middle=False,
        )
        self.layer3 = self._make_stage(
            in_channels=256,
            bottleneck_channels=256,
            conv2_out_channels=256,
            out_channels=512,
            blocks=4,
            stride=2,
            groups=groups,
            use_deformable_middle=False,
        )
        self.layer4 = self._make_stage(
            in_channels=512,
            bottleneck_channels=512,
            conv2_out_channels=512,
            out_channels=1024,
            blocks=23,
            stride=2,
            groups=groups,
            use_deformable_middle=False,
        )
        self.layer5 = self._make_stage(
            in_channels=1024,
            bottleneck_channels=1024,
            conv2_out_channels=conv5_middle_channels,
            out_channels=2048,
            blocks=3,
            stride=2,
            groups=groups,
            use_deformable_middle=True,
            deformable_middle=use_deformable_conv5,
        )

        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2048, num_classes)

    @staticmethod
    def _make_stage(
        in_channels: int,
        bottleneck_channels: int,
        conv2_out_channels: int,
        out_channels: int,
        blocks: int,
        stride: int | tuple[int, int, int],
        groups: int,
        use_deformable_middle: bool,
        deformable_middle: bool = True,
    ) -> nn.Sequential:
        layers: list[nn.Module] = [
            Bottleneck3D(
                in_channels=in_channels,
                bottleneck_channels=bottleneck_channels,
                out_channels=out_channels,
                stride=stride,
                groups=groups,
                conv2_out_channels=conv2_out_channels,
                use_deformable_middle=use_deformable_middle,
                deformable_middle=deformable_middle,
            )
        ]
        for _ in range(1, blocks):
            layers.append(
                Bottleneck3D(
                    in_channels=out_channels,
                    bottleneck_channels=bottleneck_channels,
                    out_channels=out_channels,
                    stride=1,
                    groups=groups,
                    conv2_out_channels=conv2_out_channels,
                    use_deformable_middle=use_deformable_middle,
                    deformable_middle=deformable_middle,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        if self.return_softmax:
            x = torch.softmax(x, dim=-1)
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _conv3d_flops(module: nn.Conv3d, output: torch.Tensor) -> int:
    batch_size, out_channels, out_depth, out_height, out_width = output.shape
    kernel_depth, kernel_height, kernel_width = module.kernel_size
    kernel_volume = kernel_depth * kernel_height * kernel_width
    channels_per_group = module.in_channels // module.groups
    ops_per_output = channels_per_group * kernel_volume * 2
    total = (
        batch_size * out_channels * out_depth * out_height * out_width * ops_per_output
    )
    if module.bias is not None:
        total += output.numel()
    return int(total)


def _linear_flops(module: nn.Linear, output: torch.Tensor) -> int:
    output_elements = output.numel()
    total = output_elements * module.in_features * 2
    if module.bias is not None:
        total += output_elements
    return int(total)


def _batchnorm_flops(output: torch.Tensor, affine: bool = True) -> int:
    return int(output.numel() * (2 if affine else 1))


def _relu_flops(output: torch.Tensor) -> int:
    return int(output.numel())


def _maxpool3d_flops(module: nn.MaxPool3d, output: torch.Tensor) -> int:
    kernel_depth, kernel_height, kernel_width = _triple(module.kernel_size)
    return int(output.numel() * kernel_depth * kernel_height * kernel_width)


def _adaptive_avg_pool3d_flops(
    module: nn.AdaptiveAvgPool3d, inputs: torch.Tensor, output: torch.Tensor
) -> int:
    # Average pooling over the full input volume per output element.
    return int(inputs.numel())


def _deformable_warp_flops(module: DeformableWarpConv3d, output: torch.Tensor) -> int:
    # Approximate trilinear interpolation cost per output element.
    return int(output.numel() * 16)


def _manual_flops(model: nn.Module, dummy_input: torch.Tensor) -> tuple[int, str]:
    total = 0
    hooks = []

    def register(module: nn.Module) -> None:
        nonlocal total

        def hook(
            mod: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
        ) -> None:
            nonlocal total
            if isinstance(mod, nn.Conv3d):
                total += _conv3d_flops(mod, output)
            elif isinstance(mod, nn.Linear):
                total += _linear_flops(mod, output)
            elif isinstance(mod, nn.BatchNorm3d):
                total += _batchnorm_flops(output, affine=mod.affine)
            elif isinstance(mod, nn.ReLU):
                total += _relu_flops(output)
            elif isinstance(mod, nn.MaxPool3d):
                total += _maxpool3d_flops(mod, output)
            elif isinstance(mod, nn.AdaptiveAvgPool3d):
                total += _adaptive_avg_pool3d_flops(mod, inputs[0], output)
            elif isinstance(mod, DeformableWarpConv3d):
                total += _deformable_warp_flops(mod, output)

        hooks.append(module.register_forward_hook(hook))

    for module in model.modules():
        if isinstance(
            module,
            (
                nn.Conv3d,
                nn.Linear,
                nn.BatchNorm3d,
                nn.ReLU,
                nn.MaxPool3d,
                nn.AdaptiveAvgPool3d,
                DeformableWarpConv3d,
            ),
        ):
            register(module)

    model.eval()
    with torch.no_grad():
        model(dummy_input)

    for hook in hooks:
        hook.remove()

    return total, "manual estimate"


def profile_model(
    model: nn.Module, dummy_input: torch.Tensor
) -> tuple[int, float | None, float | None, str]:
    """Return parameter count and FLOPs in MAC/operation units.

    FLOPs are computed with fvcore if available. Unsupported custom ops are
    reported in the returned note.
    """
    params = count_parameters(model)
    manual_flops, manual_note = _manual_flops(model, dummy_input)

    try:
        from fvcore.nn import FlopCountAnalysis
    except Exception as exc:  # pragma: no cover - environment dependent
        return (
            params,
            None,
            float(manual_flops),
            f"{manual_note} (fvcore unavailable: {exc})",
        )

    model.eval()
    with torch.no_grad():
        try:
            flop_analysis = FlopCountAnalysis(model, dummy_input)
            flops = float(flop_analysis.total())
            unsupported = flop_analysis.unsupported_ops()
        except Exception:
            return params, None, float(manual_flops), manual_note

    note = "fvcore"
    if unsupported:
        note = f"fvcore (unsupported ops excluded from total: {dict(unsupported)})"
    return params, flops, float(manual_flops), note


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile a Deformable ResNeXt3D-101 model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-classes", type=int, default=27)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--groups", type=int, default=32)
    parser.add_argument(
        "--conv5-middle-channels",
        type=int,
        default=2048,
        choices=[1024, 2048],
        help="Middle channel count for conv5_x; use 1024 as a fallback if the 2048 variant is undesirable.",
    )
    parser.add_argument(
        "--disable-deformable-conv5",
        action="store_true",
        help="Use a regular grouped Conv3d in conv5_x instead of the deformable wrapper.",
    )
    parser.add_argument(
        "--softmax",
        action="store_true",
        help="Apply softmax to the classifier output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = ResNeXt3D101(
        num_classes=args.num_classes,
        dropout=args.dropout,
        groups=args.groups,
        conv5_middle_channels=args.conv5_middle_channels,
        use_deformable_conv5=not args.disable_deformable_conv5,
        return_softmax=args.softmax,
    )
    dummy = torch.zeros(1, 3, 32, 112, 112)

    params, supported_flops, total_flops, note = profile_model(model, dummy)
    output = model(dummy)

    print("Deformable ResNeXt3D-101")
    print(f"  Parameters: {params:,}")
    if supported_flops is None:
        print(f"  FLOPs:      {total_flops / 1e9:.3f} GFLOPs ({note})")
    else:
        unsupported_flops = max(total_flops - supported_flops, 0.0)
        print(f"  Supported FLOPs:   {supported_flops / 1e9:.3f} GFLOPs")
        print(f"  Unsupported est.:   {unsupported_flops / 1e9:.3f} GFLOPs")
        print(f"  Hybrid total:       {total_flops / 1e9:.3f} GFLOPs ({note})")
    print(f"  Output:     {tuple(output.shape)}")


if __name__ == "__main__":
    main()
