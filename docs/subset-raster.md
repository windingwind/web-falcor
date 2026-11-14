# SubsetRaster: redraw only chosen objects

`SubsetRaster` re-rasterises a chosen set of geometry instances into its own G-buffer targets. It is
`GBufferRaster` restricted to the draws of `instanceMask`: same channels (`posW`, `normW`, `vbuffer`, `depth`, ...),
same shader and vertex math. Typical uses: selective glow or outlines on a few objects, x-ray views of hidden objects,
per-object shells. The pass keeps no state between frames.

| item | meaning |
|---|---|
| `instanceMask` (default `[]`) | geometry instance ids to draw (the id `StyleAovs` writes to `id.g` with `idSource = "instance"`); also `setInstanceMask(ids)` |
| other properties | as `GBufferRaster` (`outputSize`, `useAlphaTest`, cull options, ...) |
| input `sceneVBuffer` (optional) | the full scene's V-buffer, needed for `subsetMask` |
| output `vbuffer` | always produced (the subset ids); the other G-buffer channels stay optional |
| output `subsetMask` (optional, R32Float) | 0 = no subset geometry, 1 = subset surface visible in the full scene, 2 = subset surface hidden behind other geometry |

The subset is depth-tested only against itself, so parts hidden in the full scene are drawn too; `subsetMask`
tells them apart by comparing hit ids (instance and triangle) with `sceneVBuffer`, which is exact. Where a subset object
is visible, its subset channels equal the full G-buffer's bit for bit. `SubsetRaster` follows the main G-buffer's
camera jitter and does not install a sample pattern of its own. An empty mask draws nothing; a graph without the pass is
unchanged. `GBufferRaster` itself only gained a protected `getDraws()` hook and a flag for the jitter, with identical
behaviour.

```js
graph.addPass(createPass(device, "SubsetRaster", { instanceMask: [12, 13] }), "Subset");
graph.addEdge("GBuffer.vbuffer", "Subset.sceneVBuffer");
// Subset.subsetMask == 2: draw an x-ray tint; Subset.normW / Subset.depth: the hidden object's own surface
```

Tests: `tests/gpu/suites/subset-raster.gpu.test.ts` (bitwise match with the full raster where visible, hidden-part
classification, empty mask, main G-buffer unchanged with and without jitter; `SubsetRaster.defaultHash` prints
GBufferRaster output hashes to compare against the base commit).
