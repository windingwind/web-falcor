# Light cookies and per-instance scalar AOVs

Two optional hooks for raster lighting and per-object data. Neither keeps state between frames, and with the
hook unused the passes compile the same shaders and produce the same output as before.

## DeferredLighting: light cookie

A cookie is a texture projected from one light that multiplies that light's radiance (rgb). Typical uses:
gobos and window frames on a spot light, drifting cloud shadows on the sun.

| item | meaning |
|---|---|
| input `cookie` (optional) | any float-readable 2D texture; rgb is the multiplier, alpha is ignored |
| `cookieLight` (default -1) | index of the active light to modulate; -1 = the first directional/distant light |
| `cookieWorldSize` (default 10) | directional lights: world size of one cookie tile |
| `cookieOffset` (default [0, 0]) | directional lights: uv offset; animate it for moving patterns |

Projection (T, B = orthonormal basis of the light direction, Duff et al. 2017):
- directional/distant: `uv = (dot(p, T), dot(p, B)) / cookieWorldSize + cookieOffset`, tiled;
- spot (point light with `openingAngle` < pi/2): perspective over the cone, `uv = 0.5 + 0.5 * (dot(w, T), dot(w, B)) / (dot(w, dir) * tan(openingAngle))`
  with `w = p - light.posW`; the cone's cross-section is the disc inscribed in the [0,1]^2 square; zero outside the square
  and behind the light;
- omni point and area lights: no effect.

The lookup is bilinear with explicit texel fetches (repeat for directional, clamp for spot), so it is reproducible on
the CPU. The cookie only changes the radiance; the `visibility` output still reports shadowing alone.

```js
graph.addPass(createPass(device, "DeferredLighting", { cookieWorldSize: 25, cookieOffset: [0.1, 0] }), "Lighting");
graph.addEdge("Clouds.dst", "Lighting.cookie");   // or graph.setInput("Lighting.cookie", texture)
```

## StyleAovs: per-instance scalar (`custom` output)

`custom` (optional output, RGBA32Float) holds `instanceScalars[instance id]` in r and 1 in alpha for geometry pixels
(0 for instances past the end of the list), and 0 for the background. Set the scalars with the `instanceScalars`
property or `styleAovs.setInstanceScalars(values)`; the instance id is the one written to `id.g` with
`idSource = "instance"`. Later passes can read the AOV as a per-object weight or mask.

## Tests

`tests/gpu/suites/light-cookie.gpu.test.ts`: white cookie = bitwise unchanged radiance; directional and spot
projections against a CPU reference; the custom AOV against the scalar list, with the other StyleAovs outputs
bitwise unchanged; `LightCookie.defaultHash` prints output hashes to compare against the base commit.

## LightCookiePattern: procedural cookies

`LightCookiePattern` writes a tileable `size` x `size` cookie (RGBA32Float, rgb = lerp(1, `shadowColor`,
`strength` x occlusion)) to feed the `cookie` input above. The pattern depends only on its parameters and `time`;
the pass keeps no state.

| mode | occlusion | main parameters |
|---|---|---|
| `clouds` | fBm of periodic value noise at `frac(uv - wind x time)`, thresholded with a soft edge | `coverage`, `softness`, `scale` (cells per tile), `octaves`, `wind` (uv/s), `time` |
| `window` | one window in `rect` (uv), outer frame and mullions; light blocked outside the rect | `rect`, `panes` ([columns, rows]), `frame`, `mullion`, `softness` |
| `leaves` | sun flecks through foliage: one jittered soft disc of light per cell, optional sway | `scale`, `spotSize`, `softness`, `sway`, `swaySpeed`, `time` |

Common: `size` (default 512), `seed`, `strength` (0 = white cookie = unchanged lighting), `shadowColor`.

```js
graph.addPass(createPass(device, "LightCookiePattern", { mode: "clouds", coverage: 0.6, time: t }), "Clouds");
graph.addPass(createPass(device, "DeferredLighting", { cookieWorldSize: 200 }), "Lighting");
graph.addEdge("Clouds.dst", "Lighting.cookie");
```

Tests: `tests/gpu/suites/light-cookie-pattern.gpu.test.ts` (patterns vs a CPU reference, seamless tiling, wind
scrolling, the lighting ratio through DeferredLighting vs the CPU pattern, strength 0 = bitwise unchanged lighting,
statelessness).
