# web-falcor

WebGPU-based reimplementation of [NVIDIA Falcor](https://github.com/NVIDIAGameWorks/Falcor)
targeting 1:1 feature parity where the web platform allows. **See [DESIGN.md](DESIGN.md)**
for the framework design, the Falcor→web module mapping, and the feature parity matrix
(including features that are impossible in the browser and why).

## Layout

- `Falcor/` — pristine upstream clone (shader source of truth + native test oracle; not part of this repo)
- `tools/slang/` — Slang toolchain with WGSL backend (not part of this repo)
- `packages/falcor/` — core library, mirrors `Falcor/Source/Falcor`
- `packages/render-passes/` — render pass plugins, mirrors `Falcor/Source/RenderPasses`
- `packages/mogwai/` — browser application, mirrors `Falcor/Source/Mogwai`
- `packages/slang-compiler/` — build-time Slang→WGSL+reflection driver

## Setup

```sh
# 1. Upstream Falcor clone + native deps (oracle build)
git clone https://github.com/NVIDIAGameWorks/Falcor.git && (cd Falcor && ./setup.sh)
cmake --preset linux-gcc -S Falcor && cmake --build Falcor/build/linux-gcc -j16

# 2. Slang toolchain (WGSL backend)
mkdir -p tools/slang && curl -L https://github.com/shader-slang/slang/releases/download/v2026.12.2/slang-2026.12.2-linux-x86_64.tar.gz | tar xz -C tools/slang

# 3. Web workspace
npm install
npm run shaders    # Slang -> WGSL + reflection JSON
npm run typecheck
npm run dev        # Mogwai dev server (needs a WebGPU browser)
```
