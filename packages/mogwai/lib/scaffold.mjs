/**
 * Render pass templates, shared by `npm run new:pass` (in this repo) and `web-falcor new` (the npm CLI).
 * Modes: "in-tree" (packages/render-passes), "repo-plugin" (plugins/ of a checkout), "standalone" (its own project).
 */

import { readFileSync } from "node:fs";

/** The published package version, for a standalone project's dependencies. */
const kVersion = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")).version;

export const isPassName = (name) => /^[A-Z][A-Za-z0-9]*$/.test(name);
export const kebabCase = (name) => name.replace(/([a-z0-9])([A-Z])/g, "$1-$2").toLowerCase();

/** [file, text] pairs for a pass `name` in `mode`. */
export function passFiles(name, mode) {
    const plugin = mode !== "in-tree";
    const files = [
        [`${name}.ts`, passTs(name, plugin)],
        [`${name}.cs.slang`, passSlang(name)],
        [`${name}.graph.js`, graphJs(name)],
    ];
    if (plugin) files.push(["README.md", readme(name, mode)], ["package.json", packageJson(name, mode)], ["vite.config.js", viteConfig(name)]);
    if (mode === "standalone") files.push(["tsconfig.json", tsconfig()], [".gitignore", "node_modules/\ndist/\n"]);
    return files;
}

function passTs(name, plugin) {
    // In-tree passes use native's RenderPasses/<Pass>/ layout; plugins inline their shader as a string module.
    const shaderKey = `${plugin ? "Plugins" : "RenderPasses"}/${name}/${name}.cs.slang`;
    const shaderImport = plugin ? `\n// Vite inlines the shader source; it compiles as a string module, so no shader list entry is needed.\nimport shaderSource from "./${name}.cs.slang?raw";\n` : "";
    const shaderDesc = plugin ? `{ modules: [{ sources: [{ string: shaderSource, path: "${shaderKey}" }] }] }` : `{ path: "${shaderKey}" }`;
    return `${plugin ? `/// <reference types="vite/client" />\n` : ""}/**
 * ${name}: scales its input by a factor in a compute shader.
 * Scaffolded by web-falcor; see https://github.com/windingwind/web-falcor/blob/main/docs/extending.md.
 */

import {
    ComputePass,
    Properties,
    RenderData,
    RenderPass,
    RenderPassReflection,
    ResourceBindFlags,
    ResourceFormat,
    registerRenderPass,
    type CompileData,
    type Device,
    type RenderContext,
    type UIWidgets,
} from "@web-falcor/falcor";
${shaderImport}
export class ${name} extends RenderPass {
    private scale = 1;
    private pass: ComputePass | null = null;

    constructor(device: Device, props: Properties) {
        super(device);
        this.scale = props.get("scale", 1);
    }

    override getProperties(): Properties {
        return new Properties({ scale: this.scale });
    }

    override renderUI(ui: UIWidgets): void {
        ui.slider("Scale", this.scale, 0, 4, 0.01, (v) => (this.scale = v));
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        r.addInput("src", "Input color").bindFlags(ResourceBindFlags.ShaderResource);
        r.addOutput("dst", "Scaled color")
            .texture2D(w, h)
            .format(ResourceFormat.RGBA32Float)
            .bindFlags(ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource);
        return r;
    }

    override execute(ctx: RenderContext, renderData: RenderData): void {
        const src = renderData.getTexture("src");
        const dst = renderData.getTexture("dst");
        if (!src || !dst) return;

        this.pass ??= ComputePass.create(this.device, ${shaderDesc});
        const root = this.pass.getRootVar();
        root["CB"]["frameDim"] = [dst.width, dst.height];
        root["CB"]["scale"] = this.scale;
        root["src"] = src;
        root["dst"] = dst;
        this.pass.execute(ctx, dst.width, dst.height);
    }
}

registerRenderPass("${name}", (device, props) => new ${name}(device, props), { desc: "Scales its input by a factor" });
`;
}

function passSlang(name) {
    return `/** ${name}: dst = scale * src. Compiled Slang -> WGSL in the browser. */

cbuffer CB
{
    uint2 frameDim;
    float scale;
}

Texture2D<float4> src;
RWTexture2D<float4> dst;

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    const uint2 pixel = dispatchThreadId.xy;
    if (any(pixel >= frameDim))
        return;
    dst[pixel] = scale * src[pixel];
}
`;
}

function graphJs(name) {
    return `// Path tracer -> ToneMapper -> ${name}; open in Mogwai with ?script=<this file>.
// A JS render-graph script: its context has what Python's \`from falcor import *\` gives (plus \`m\`).
export default function ({ m, RenderGraph, createPass }) {
    const g = RenderGraph("${name}");
    g.addPass(createPass("VBufferRT", { samplePattern: "Stratified", sampleCount: 16 }), "VBufferRT");
    g.addPass(createPass("PathTracer", { samplesPerPixel: 1 }), "PathTracer");
    g.addPass(createPass("AccumulatePass", { enabled: true, precisionMode: "Single" }), "AccumulatePass");
    g.addPass(createPass("ToneMapper", { autoExposure: false, exposureCompensation: 0 }), "ToneMapper");
    g.addPass(createPass("${name}", { scale: 1 }), "${name}");
    g.addEdge("VBufferRT.vbuffer", "PathTracer.vbuffer");
    g.addEdge("VBufferRT.viewW", "PathTracer.viewW");
    g.addEdge("VBufferRT.mvec", "PathTracer.mvec");
    g.addEdge("PathTracer.color", "AccumulatePass.input");
    g.addEdge("AccumulatePass.output", "ToneMapper.src");
    g.addEdge("ToneMapper.dst", "${name}.src");
    g.markOutput("${name}.dst");
    m.addGraph(g);
}
`;
}

function packageJson(name, mode) {
    const pkg = {
        name: kebabCase(name),
        version: "0.1.0",
        type: "module",
        description: `${name}: a web-falcor render pass plugin`,
        files: ["dist", "*.graph.js", "README.md"],
        // Read by `web-falcor dev` to open the viewer on this plugin.
        webFalcor: { scene: "test_scenes/cornell_box.pyscene", plugin: `${name}.ts`, script: `${name}.graph.js` },
        scripts: mode === "standalone" ? { dev: "web-falcor dev", build: "vite build", typecheck: "tsc --noEmit" } : { build: "vite build" },
        peerDependencies: { "@web-falcor/falcor": "*" },
    };
    if (mode === "standalone") {
        pkg.devDependencies = {
            "@web-falcor/falcor": `^${kVersion}`,
            "@web-falcor/mogwai": `^${kVersion}`,
            "@webgpu/types": "^0.1.60",
            typescript: "^5.8.0",
            vite: "^6.3.0",
        };
    }
    return JSON.stringify(pkg, null, 2) + "\n";
}

function viteConfig(name) {
    return `import { defineConfig } from "vite";

// Builds dist/${name}.js, loadable in any web-falcor viewer with ?plugin=<its URL>; the viewer provides @web-falcor/*.
export default defineConfig({
    build: {
        target: "es2022",
        lib: { entry: "${name}.ts", formats: ["es"], fileName: () => "${name}.js" },
        rollupOptions: { external: [/^@web-falcor\\//] },
    },
});
`;
}

function tsconfig() {
    return (
        JSON.stringify(
            {
                compilerOptions: {
                    target: "ES2022",
                    module: "ESNext",
                    moduleResolution: "bundler",
                    lib: ["ES2022", "DOM", "DOM.Iterable"],
                    types: ["vite/client", "@webgpu/types"],
                    strict: true,
                    noImplicitOverride: true,
                    skipLibCheck: true,
                    noEmit: true,
                },
                include: ["*.ts"],
            },
            null,
            2,
        ) + "\n"
    );
}

function readme(name, mode) {
    const develop =
        mode === "standalone"
            ? `\`\`\`sh
npm install
npm run dev     # serves the prebuilt web-falcor viewer with this plugin loaded
\`\`\`

Edit \`${name}.ts\` and \`${name}.cs.slang\`, then reload the page.`
            : `From a web-falcor checkout (after its quick start):

\`\`\`sh
git clone <this repo's URL> plugins/${name}
npm run dev
\`\`\`

Then open
<http://localhost:5173/?scene=test_scenes/cornell_box.pyscene&script=/plugins/${name}/${name}.graph.js&plugin=/plugins/${name}/${name}.ts>.`;
    return `# ${name}

A [web-falcor](https://github.com/windingwind/web-falcor) render pass plugin.

## Develop

${develop}

## Share

\`npm run build\` writes \`dist/${name}.js\`. Host it and \`${name}.graph.js\` anywhere that allows CORS (e.g. GitHub Pages),
then open the online viewer with both URLs:

\`\`\`text
https://windingwind.github.io/web-falcor/?plugin=<url of ${name}.js>&script=<url of ${name}.graph.js>
\`\`\`

To use the pass in your own graph, load the plugin with \`?plugin=\` and create it with
\`createPass("${name}", { scale: 1 })\`.
`;
}
