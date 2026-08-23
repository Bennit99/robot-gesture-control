import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add project root to sys.path so we can import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.model_loader import load_pytorch_model


def benchmark_pytorch(model_path, input_shape, num_runs=1000, warmup=100, device="cpu"):
    print(f"[PyTorch] Loading {model_path} on {device}...")
    model = load_pytorch_model(model_path, device=device)
    
    # Create dummy input
    dummy_input = torch.randn(1, *input_shape).to(device)
    
    # Warmup
    print(f"[PyTorch] Warming up for {warmup} iterations...")
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(dummy_input)
            
    if device != "cpu":
        torch.cuda.synchronize()
        
    print(f"[PyTorch] Benchmarking {num_runs} iterations...")
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        if device != "cpu":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000) # ms
        
    return times

def benchmark_tensorrt(engine_path, input_shape, num_runs=1000, warmup=100):
    print(f"[TensorRT] Loading {engine_path}...")
    try:
        import pycuda.autoinit
        import pycuda.driver as cuda
        import tensorrt as trt
    except ImportError:
        print("[TensorRT] tensorrt or pycuda not installed. Skipping TensorRT benchmark.")
        return None
        
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        
    context = engine.create_execution_context()
    
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    
    # TRT 8.5+ API compatibility
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        shape = engine.get_tensor_shape(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        
        # TRT bindings can sometimes have dynamic batch sizes (e.g., -1). We default to 1.
        shape = [1 if d == -1 else d for d in shape]
        size = trt.volume(shape)
        
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        
        if is_input:
            # fill with random data
            np.copyto(host_mem, np.random.randn(*shape).astype(dtype))
            inputs.append({'host': host_mem, 'device': device_mem, 'name': name})
        else:
            outputs.append({'host': host_mem, 'device': device_mem, 'name': name})
            
    # Warmup
    print(f"[TensorRT] Warming up for {warmup} iterations...")
    for _ in range(warmup):
        for inp in inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], stream)
            
        try:
            for i, binding in enumerate(bindings):
                name = engine.get_tensor_name(i)
                context.set_tensor_address(name, binding)
            context.execute_async_v3(stream_handle=stream.handle)
        except AttributeError:
            # Older TRT fallback
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            
        for out in outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], stream)
        stream.synchronize()
        
    print(f"[TensorRT] Benchmarking {num_runs} iterations...")
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        
        for inp in inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], stream)
            
        try:
            context.execute_async_v3(stream_handle=stream.handle)
        except AttributeError:
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            
        for out in outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], stream)
        stream.synchronize()
        
        times.append((time.perf_counter() - start) * 1000) # ms
        
    return times

def benchmark_onnx(onnx_path, input_shape, num_runs=1000, warmup=100):
    print(f"[ONNX] Loading {onnx_path}...")
    import onnxruntime as ort
    
    # Try GPU first, fallback to CPU
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        session = ort.InferenceSession(onnx_path, providers=providers)
    except Exception as e:
        print(f"[ONNX] Failed to load session: {e}")
        return None
        
    input_name = session.get_inputs()[0].name
    # Create dummy input matching the expected shape and dtype
    dummy_input = np.random.randn(1, *input_shape).astype(np.float32)
    
    print(f"[ONNX] Warming up for {warmup} iterations...")
    for _ in range(warmup):
        _ = session.run(None, {input_name: dummy_input})
        
    print(f"[ONNX] Benchmarking {num_runs} iterations...")
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input})
        times.append((time.perf_counter() - start) * 1000) # ms
        
    return times

def main():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs TensorRT Latency")
    parser.add_argument("--ckpt", type=str, default="models/gru-12_ml9.ckpt")
    parser.add_argument("--onnx", type=str, default="models/gru-12_ml9.onnx")
    parser.add_argument("--engine", type=str, default="models/gru-12_ml9.engine")
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    args = parser.parse_args()
    
    # Jester architecture default
    seq_len = 40
    input_dim = 63
    input_shape = (seq_len, input_dim)
    
    results = {}
    
    # 1. PyTorch CPU
    pt_cpu_times = benchmark_pytorch(args.ckpt, input_shape, device="cpu")
    if pt_cpu_times:
        results["PyTorch_CPU"] = {
            "mean_ms": np.mean(pt_cpu_times),
            "std_ms": np.std(pt_cpu_times),
            "median_ms": np.median(pt_cpu_times),
            "min_ms": np.min(pt_cpu_times),
            "max_ms": np.max(pt_cpu_times)
        }
        
    # 2. PyTorch GPU
    if torch.cuda.is_available():
        pt_gpu_times = benchmark_pytorch(args.ckpt, input_shape, device="cuda")
        if pt_gpu_times:
            results["PyTorch_GPU"] = {
                "mean_ms": np.mean(pt_gpu_times),
                "std_ms": np.std(pt_gpu_times),
                "median_ms": np.median(pt_gpu_times),
                "min_ms": np.min(pt_gpu_times),
                "max_ms": np.max(pt_gpu_times)
            }
            
    # 3. ONNX
    if os.path.exists(args.onnx):
        onnx_times = benchmark_onnx(args.onnx, input_shape)
        if onnx_times:
            results["ONNX_Runtime"] = {
                "mean_ms": np.mean(onnx_times),
                "std_ms": np.std(onnx_times),
                "median_ms": np.median(onnx_times),
                "min_ms": np.min(onnx_times),
                "max_ms": np.max(onnx_times)
            }
            
    # 4. TensorRT
    if os.path.exists(args.engine):
        trt_times = benchmark_tensorrt(args.engine, input_shape)
        if trt_times:
            results["TensorRT"] = {
                "mean_ms": np.mean(trt_times),
                "std_ms": np.std(trt_times),
                "median_ms": np.median(trt_times),
                "min_ms": np.min(trt_times),
                "max_ms": np.max(trt_times)
            }
    else:
        print(f"[TensorRT] Engine file not found at {args.engine}. Please ensure it is compiled first.")
        
    # Print and save
    print("\n" + "="*50)
    print("BENCHMARK RESULTS (Milliseconds per Frame)")
    print("="*50)
    
    base_latency = None
    if "PyTorch_GPU" in results:
        base_latency = results["PyTorch_GPU"]["mean_ms"]
    elif "PyTorch_CPU" in results:
        base_latency = results["PyTorch_CPU"]["mean_ms"]
        
    for name, metrics in results.items():
        print(f"--- {name} ---")
        print(f"  Mean   : {metrics['mean_ms']:.4f} ms")
        print(f"  Median : {metrics['median_ms']:.4f} ms")
        print(f"  StdDev : {metrics['std_ms']:.4f} ms")
        print(f"  Min/Max: {metrics['min_ms']:.4f} / {metrics['max_ms']:.4f} ms")
        
        if base_latency and name == "TensorRT":
            speedup = base_latency / metrics['mean_ms']
            print(f"  --> SPEEDUP vs PyTorch: {speedup:.2f}x")
            results[name]["speedup_vs_pytorch"] = speedup
            
    print("="*50)
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {args.output}")

if __name__ == "__main__":
    main()
