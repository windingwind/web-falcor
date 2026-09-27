# @web-falcor/mogwai

Mogwai, [NVIDIA Falcor](https://github.com/NVIDIAGameWorks/Falcor)'s interactive viewer, prebuilt for the browser
by [web-falcor](https://github.com/windingwind/web-falcor). It needs a browser with WebGPU (Chrome or Edge).

```sh
npx @web-falcor/mogwai                      # serves the viewer at http://localhost:5173/
npx @web-falcor/mogwai --plugins ./plugins  # also serves ./plugins at /plugins/
npx @web-falcor/mogwai --media ./media      # serves your scenes at /Falcor/media/
```

Write a plugin without a web-falcor checkout:

```sh
npx @web-falcor/mogwai new MyPass && cd my-pass && npm install && npm run dev
```

- Open a scene with `?scene=<path under /Falcor/media/>` and a render graph with `?script=<url of a .py graph>`.
- Load a prebuilt plugin with `?plugin=<url of its .js>`; the parameter can be repeated.
- `npx @web-falcor/mogwai assets <dir>` copies the runtime assets (shaders, the Slang compiler, Pyodide) that apps
  built on `@web-falcor/falcor` need; see [Using the npm packages](https://github.com/windingwind/web-falcor/blob/main/docs/npm.md).

The package ships Falcor's shaders (BSD-3) and the default Cornell box scene. The NRD and RTXDI shader sources are
not included; the viewer fetches them from NVIDIA's GitHub repositories when a pass needs them.
