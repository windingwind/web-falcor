# Stylized focus and selective bloom

Two stateless post passes that steer attention in a frame. Both read HDR colour and add nothing when disabled
(identity: the output equals `src` bit for bit).

## StylizedFocus

A focus weight f (1 = in focus) is computed per pixel; the out-of-focus amount a = `strength` x (1 - f) drives blur
and desaturation. In-focus pixels (a = 0) are copied unchanged.

| mode | focus weight | inputs |
|---|---|---|
| `depth` | 1 - smoothstep(`focusRange`, `focusRange` + `falloff`, \|distance to camera - `focusDistance`\|) | `posW` |
| `point` | 1 - smoothstep(`focusRange`, `focusRange` + `falloff`, \|posW - `focusPoint`\|) | `posW` |
| `screen` | same on the distance to `screenCenter` (uv), in frame heights | — |
| `ids` | 1 on visible pixels of the listed ids (raw id in `id.g`, at most 32) | `id` (StyleAovs.id) |
| `mask` | saturate(mask.r) | `mask` |

Background pixels get `backgroundFocus` in the depth and point modes. Blur: two separable Gaussians of `src` at
sigma = `blurRadius` x frame height and sigma / 3, blended by a; desaturation: lerp towards Rec. 709 luma by
a x `desaturate`. Optional output `focus` = f.

The approach follows Cole et al. 2006, "Directing Gaze in 3D Models with Stylized Focus" (emphasis falling off with
the distance to a focus controls the amount of abstraction); only blur and desaturation are implemented.

## SelectiveBloom

Bloom emitted by selected objects only. The selection is `ids` (at most 32, matched against the raw id in `id.g`;
use StyleAovs with `idSource = "instance"`), optionally multiplied by a `mask` input (r); `invert` swaps it. Only
visible pixels of the selection emit; the glow spreads over neighbouring pixels.

Steps: half-resolution 2x2 prefilter of `src` x selection with a soft-knee `threshold`/`knee`; separable Gaussian with
sigma = `radius` x half height; `dst = src + intensity x tint x glow`. Optional output `glow`.
`intensity` 0, or an empty selection, copies `src`.

The idea of masking the bloom input by an object selection follows pmndrs/postprocessing's SelectiveBloomEffect
(zlib licence, Copyright (c) 2015 Raoul van Rüschen); the code is independent.

```js
graph.addPass(createPass(device, "StyleAovs", { idSource: "instance" }), "Aovs");
graph.addPass(createPass(device, "SelectiveBloom", { ids: [12, 40], intensity: 2, radius: 0.03 }), "Glow");
graph.addPass(createPass(device, "StylizedFocus", { mode: "ids", ids: [12, 40], blurRadius: 0.01 }), "Focus");
graph.addEdge("GBuffer.vbuffer", "Aovs.vbuffer");
graph.addEdge("Lighting.color", "Focus.src");
graph.addEdge("Aovs.id", "Focus.id");
graph.addEdge("Focus.dst", "Glow.src");
graph.addEdge("Aovs.id", "Glow.id");
```

## Tests

`tests/gpu/suites/stylized-focus.gpu.test.ts`: both passes against a CPU reference (all focus modes, id lists,
invert, mask, threshold/knee, odd sizes); unselected objects add no glow; StyleAovs ids on a test scene; identity
cases bit for bit; statelessness.
