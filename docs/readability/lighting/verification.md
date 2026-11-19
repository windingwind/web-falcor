# Clarify lighting, visibility and atmosphere effects: verification

Lighting and atmosphere wrappers used terse private fields and dense setup expressions. This change documents their contracts, types and consistently names private storage, and expands the existing resource setup and shader statements. It covers raster lighting and visibility, SSAO, SSR, SSGI, sky and clouds, fog, rain, ripples and style shading.

The change was independently based on upstream `main` at `4baecd99ec8fd38094b1f8177a90d66205f5a9e0`. The tested source commit is `c2aeffb6997b40d68555e5644551c31e6ae1d34d`. The following evidence commit only adds this document and `results.json`; it does not change the tested source or tests.

## Results

| Check | Main | This group | Comparison |
| --- | --- | --- | --- |
| Changed TypeScript runtime structure | 20 files | 20 files | Equal after the normalization below; one shader-inspection adapter exception |
| Changed shader token sequences | 9 files | 9 files | Identical |
| Full GPU suite | 371 pass / 236 fail / 144 skip | 371 pass / 236 fail / 144 skip | All 751 case statuses identical; no new failures |
| Unit suite | 431 pass / 8 skip | 431 pass / 8 skip | Same outcomes; exit 0 |
| Workspace type checks | Pass | Pass | Exit 0 |
| Gallery texture readbacks | 214 outputs | 214 outputs | Every byte identical |

The 236 GPU failures above also occur on main on this stack. They are retained in the evidence rather than treated as passing tests. No assertions, tolerances or failure expectations were relaxed.

## Static method

Changed TypeScript is emitted as ES2022 JavaScript with TypeScript. The comparison reverses the recorded private/local renames, removes redundant parentheses, and splits multi-variable declarations while retaining `const`/`let`. It compares AST node kinds, identifiers, literal values, operators and declaration flags. Types and comments are erased by the compiler. The exact per-file normalized hashes and rename map are in [results.json](results.json).

Changed Slang files are compared as ordered non-comment tokens, retaining whole identifiers, numeric and string literals, operators and punctuation. Preprocessor directives and continuation lines also remain identical. The shader edits only expand existing statements. Files outside the listed source changes are identical to main.

The lighting group also adapts `wgslHashes()` in the existing shadow-clip test to inspect either `mPass`/`pass` or `mState`/`state`. This function is the sole exception to AST equality. Every graph setup, render call, assertion and tolerance outside that function is included in the comparison. The suite's printed default-output and compiled-WGSL hash lines match between the pair.

Formatting uses Prettier 3.10.1 with four-space indentation and a 140-column width. The core checks cover only the edited helper/initializer, preserving the surrounding existing file layout. Other changed TypeScript files are checked in full. `git diff --check` passes. The repository has no standalone general lint command.

## Paired execution

Main and this group were run on the same GPU, driver, dependencies and browser configuration. The shared GPU lock is released between completed suites to allow other work to interleave. Each run starts a fresh browser. The entire registered GPU suite and the existing workspace unit/type checks run without changing their assertions or tolerance settings.

- GPU and driver: NVIDIA TITAN RTX, 610.57.04.
- Node: v22.13.1; TypeScript: 5.9.3; Playwright: 1.61.1; Chromium: 149.0.7827.55.
- Browser: headless Chromium with hardware Vulkan; exact launch flags are recorded in `results.json`.
- Case-status SHA-256, main and this group: `3624796e7038abefdb3152f17e24a234325d2fe7f8b09e377880ed3936c536d2`.
- Printed default-output/compiled-WGSL hash lines: 19 matching lines.

## Gallery method

Gallery stack: NVIDIA GeForce RTX 4080 SUPER, 610.57.04; Node v22.13.1. Main and this group use the same gallery stack.

The same 50 contributed passes are rendered at the existing BistroExterior and SunTemple scene cameras. Both runs use a 256 x 256 graph, centre sampling, the same input graph and parameters, and four graph executions (the first execution followed by three further frames). All 107 texture outputs per camera are read back as raw bytes; preview conversion is excluded from the comparison. The graph includes the shared G-buffer, lighting, shadow, style AOV and depth-pyramid inputs used by the effects.

Scene loading is checked before rendering: both scenes contain more than 1,000 triangles, and the position buffer confirms visible geometry. The per-scene triangle counts, visible-geometry coverage and all 214 before/after SHA-256 values are in [results.json](results.json). Equality is exact, without an image tolerance.
