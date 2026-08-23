# Frontend CSS Notes


## Tailwind Local Setup (No Internet at Runtime)

Files used:
- `tailwind.input.css`: Tailwind entry file with `@tailwind` directives and custom layers.
- `tailwind.config.js`: content scanning config for used classes.
- `tailwind.local.css`: generated output used by `index.html`.

`index.html` should reference only:

```html
<link rel="stylesheet" href="./tailwind.local.css" />
```

## Rebuild CSS After UI Changes

Rebuild whenever you:
- add/remove Tailwind classes in `index.html` or `js/**/*.js`
- change `tailwind.input.css`
- change `tailwind.config.js`

From project root:

```powershell
Set-Location .\plot\streaming_web_server\frontend
C:\Users\stu245938\Desktop\robot-gesture-control\tools\tailwindcss-v3.exe -i .\tailwind.input.css -o .\tailwind.local.css -c .\tailwind.config.js -m
```

Quick checks:

```powershell
Get-Item .\tailwind.local.css | Select-Object Name,Length,LastWriteTime
```

## Practical Hints

- Keep custom component styles in `tailwind.input.css` (`@layer components` / `@layer utilities`).
- Keep icon mappings centralized in `fontawesome.local.css`.
- If a class seems missing in output CSS, ensure it appears in files matched by `tailwind.config.js` `content`.


## Local Icons (Font Awesome Replacement)


This frontend uses a local subset icon pack compatible with the existing
`fa-*` classes:
- `fontawesome.local.css`
- `icons/fa/*.svg`

`index.html` should reference:

```html
<link rel="stylesheet" href="./fontawesome.local.css" />
```

Currently included icon subset (only used ones):
- `fa-xmark`
- `fa-spinner`
- `fa-moon`
- `fa-server`
- `fa-circle-info`
- `fa-stop`
- `fa-circle-notch`
- `fa-gear`
- `fa-play`
- `fa-circle-xmark`
- `fa-triangle-exclamation`

To add a new icon later:
1. Add `icons/fa/<name>.svg`
2. Add `.fa-<name>::before` mapping in `fontawesome.local.css`
3. Use `class="fa-solid fa-<name>"` in HTML/JS
- If styles look stale in browser, hard refresh and clear service worker cache.
