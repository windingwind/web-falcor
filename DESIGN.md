# web-falcor — Framework Design

A WebGPU-based reimplementation of [NVIDIA Falcor 8.0](https://github.com/NVIDIAGameWorks/Falcor)
targeting **exact 1:1 feature parity wherever the web platform allows**, with every
gap explicitly marked and explained.

Reference: upstream clone at `./Falcor` (commit `eb540f67`, built natively at
`Falcor/build/linux-gcc` for ground-truth comparison), research fork at `../Falcor`.

---

This document is an index. The design is split across [`docs/`](docs/) so each
part stays readable; the original section numbers (§1–§11) are **preserved
across the files**, so every `§N` cross-reference in the text resolves via the
map below.

## Contents

| Sections | Document | Covers |
|---|---|---|
| §1–§3 | [docs/architecture.md](docs/architecture.md) | Goals & ground rules · WebGPU-vs-WebGL2 platform decision · architecture overview + repo layout |
| §4–§5 | [docs/shader-system.md](docs/shader-system.md) | Slang→WGSL compilation pipeline · reflection-driven binding · WebFalcor platform shims · software ray tracing (no RT hardware) |
| §6 | [docs/module-mapping.md](docs/module-mapping.md) | Module-by-module Falcor→web mapping (`Core`, `Scene`, `Rendering`, `RenderGraph`, UI, scripting, `Utils`, `DiffRendering`) |
| §7 | [docs/testing.md](docs/testing.md) | Testing strategy · verified oracle results (web vs native DXR) · upstream image-test graph pass-rate |
| §8–§9 | [docs/parity-matrix.md](docs/parity-matrix.md) | Feature parity matrix (platform/core, all 31 render passes, ecosystem) · known behavioral divergences |
| §10–§11 | [docs/roadmap.md](docs/roadmap.md) | Roadmap (M0–M8) · resolved design questions |

## Status markers

The ✅ / 🟡 / 🔶 / ❌ / 🟠 markers used throughout the parity matrix and mapping
tables are defined in [architecture.md §1](docs/architecture.md#1-goals-and-ground-rules).
