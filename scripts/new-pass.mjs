#!/usr/bin/env node
/**
 * Scaffolds a render pass: a compute pass (TS + Slang) and a Mogwai graph script that runs it.
 *
 *   npm run new:pass -- <Name>            in-tree, under packages/render-passes/src/<Name>/
 *   npm run new:pass -- <Name> --plugin   out-of-tree plugin, under plugins/<Name>/ (loaded with ?plugin=)
 */

import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync, appendFileSync } from "node:fs";
import { join, relative, resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { refreshShaderList } from "../packages/slang-compiler/bin/repo-shader-list.js";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const passesSrc = join(repoRoot, "packages/render-passes/src");

function usage(msg) {
    if (msg) console.error(`error: ${msg}\n`);
    console.error("usage: npm run new:pass -- <Name> [--plugin]\n  <Name>: PascalCase pass type, e.g. MyBlur");
    process.exit(msg ? 1 : 0);
}

const args = process.argv.slice(2);
if (args.includes("-h") || args.includes("--help")) usage();
const plugin = args.includes("--plugin");
const unknown = args.filter((a) => a.startsWith("-") && a !== "--plugin");
if (unknown.length > 0) usage(`unknown option ${unknown[0]}`);
const names = args.filter((a) => !a.startsWith("-"));
if (names.length !== 1) usage("give exactly one pass name");
const name = names[0];
if (!/^[A-Z][A-Za-z0-9]*$/.test(name)) usage(`'${name}' is not a PascalCase identifier`);

/** Pass types already registered in-tree (registerRenderPass("X", ...)). */
function registeredPassTypes() {
    const types = new Set();
    const walk = (dir) => {
        for (const e of readdirSync(dir, { withFileTypes: true })) {
            const p = join(dir, e.name);
            if (e.isDirectory()) walk(p);
            else if (e.name.endsWith(".ts")) for (const m of readFileSync(p, "utf8").matchAll(/registerRenderPass\(\s*"([^"]+)"/g)) types.add(m[1]);
        }
    };
    walk(passesSrc);
    return types;
}
if (registeredPassTypes().has(name)) usage(`a pass named '${name}' is already registered in packages/render-passes`);

const outDir = plugin ? join(repoRoot, "plugins", name) : join(passesSrc, name);
if (existsSync(outDir)) usage(`${relative(repoRoot, outDir)} already exists`);

// Shader path: in-tree passes use native's RenderPasses/<Pass>/ layout; plugins inline their shader as a string module.
const shaderKey = `${plugin ? "Plugins" : "RenderPasses"}/${name}/${name}.cs.slang`;
const shaderImport = plugin ? `\n// Vite inlines the shader source; it compiles as a string module, so no shader list entry is needed.\nimport shaderSource from "./${name}.cs.slang?raw";\n` : "";
const shaderDesc = plugin ? `{ modules: [{ sources: [{ string: shaderSource, path: "${shaderKey}" }] }] }` : `{ path: "${shaderKey}" }`;

const passTs = `${plugin ? `/// <reference types="vite/client" />\n` : ""}/**
 * ${name}: scales its input by a factor in a compute shader.
 * Scaffolded by \`npm run new:pass\`; see docs/extending.md.
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

const passSlang = `/** ${name}: dst = scale * src. Compiled Slang -> WGSL in the browser. */

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

const graphPy = `from falcor import *

# Path tracer -> ToneMapper -> ${name}; open in Mogwai with ?script=<this file>.
def render_graph_${name}():
    g = RenderGraph("${name}")
    g.addPass(createPass("VBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16}), "VBufferRT")
    g.addPass(createPass("PathTracer", {'samplesPerPixel': 1}), "PathTracer")
    g.addPass(createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'}), "AccumulatePass")
    g.addPass(createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0}), "ToneMapper")
    g.addPass(createPass("${name}", {'scale': 1.0}), "${name}")
    g.addEdge("VBufferRT.vbuffer", "PathTracer.vbuffer")
    g.addEdge("VBufferRT.viewW", "PathTracer.viewW")
    g.addEdge("VBufferRT.mvec", "PathTracer.mvec")
    g.addEdge("PathTracer.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.addEdge("ToneMapper.dst", "${name}.src")
    g.markOutput("${name}.dst")
    return g

${name} = render_graph_${name}()
try: m.addGraph(${name})
except NameError: None
`;

const pluginPackageJson = JSON.stringify(
    {
        name: name.replace(/([a-z0-9])([A-Z])/g, "$1-$2").toLowerCase(),
        version: "0.1.0",
        type: "module",
        description: `${name}: a web-falcor render pass plugin`,
        files: ["dist", "*.py", "README.md"],
        scripts: { build: "vite build" },
        peerDependencies: { "@web-falcor/falcor": "*" },
    },
    null,
    2,
) + "\n";

const pluginViteConfig = `import { defineConfig } from "vite";

// Builds dist/${name}.js, loadable in any web-falcor viewer with ?plugin=<its URL>; the viewer provides @web-falcor/*.
export default defineConfig({
    build: {
        lib: { entry: "${name}.ts", formats: ["es"], fileName: () => "${name}.js" },
        rollupOptions: { external: [/^@web-falcor\\//] },
    },
});
`;

const readme = `# ${name}

A [web-falcor](https://github.com/windingwind/web-falcor) render pass plugin.

## Try it

Build it with \`npm run build\`, host \`dist/${name}.js\` and \`${name}.py\` anywhere that allows CORS (e.g. GitHub Pages),
then open the online viewer with both URLs:

\`\`\`text
https://windingwind.github.io/web-falcor/?plugin=<url of ${name}.js>&script=<url of ${name}.py>
\`\`\`

## Develop

From a web-falcor checkout (after its quick start):

\`\`\`sh
git clone <this repo's URL> plugins/${name}
npm run dev
\`\`\`

Then open
<http://localhost:5173/?scene=test_scenes/cornell_box.pyscene&script=/plugins/${name}/${name}.py&plugin=/plugins/${name}/${name}.ts>.

To use the pass in your own graph, load the plugin with \`?plugin=\` and create it with
\`createPass("${name}", {'scale': 1.0})\`.
`;

mkdirSync(outDir, { recursive: true });
const written = [
    [`${name}.ts`, passTs],
    [`${name}.cs.slang`, passSlang],
    [`${name}.py`, graphPy],
    ...(plugin
        ? [
              ["README.md", readme],
              ["package.json", pluginPackageJson],
              ["vite.config.js", pluginViteConfig],
          ]
        : []),
].map(([file, text]) => {
    writeFileSync(join(outDir, file), text);
    return relative(repoRoot, join(outDir, file));
});

if (!plugin) {
    appendFileSync(join(passesSrc, "index.ts"), `export * from "./${name}/${name}.js";\n`);
    written.push("packages/render-passes/src/index.ts (export added)");
    refreshShaderList();
    written.push("packages/falcor/shaders/generated/shader-file-list.json (shader registered)");
}

const dir = "/" + relative(repoRoot, outDir).replaceAll("\\", "/");
const url = `http://localhost:5173/?scene=test_scenes/cornell_box.pyscene&script=${dir}/${name}.py${plugin ? `&plugin=${dir}/${name}.ts` : ""}`;
console.log(`Created ${plugin ? "plugin" : "in-tree"} pass ${name}:\n${written.map((f) => `  ${f}`).join("\n")}`);
console.log(`\nNext:\n  npm run dev\n  open ${url}`);
if (plugin) console.log(`\nplugins/ is gitignored: publish ${name} from its own repo (cd ${relative(repoRoot, outDir)} && git init).`);
