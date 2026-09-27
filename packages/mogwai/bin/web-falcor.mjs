#!/usr/bin/env node
/**
 * Serves the prebuilt Mogwai viewer (`npx @web-falcor/mogwai`), plus local plugins and scenes.
 *
 *   web-falcor [--port 5173] [--plugins <dir>] [--media <dir>] [--site <dir>]
 *   web-falcor new <Name>        scaffold a plugin project in ./<name>/ (a compute pass, its shader and a graph)
 *   web-falcor dev [--port 5173] develop the plugin in the current project against the prebuilt viewer
 *   web-falcor assets <dir>      copy the runtime assets for @web-falcor/falcor apps into <dir>
 *   web-falcor py2js <file-or-dir>... [--out <dir>] [--root <dir>] [--check]   convert Python scripts to JS
 *
 * `--plugins` is served at /plugins/ (load one with ?plugin=/plugins/<Name>/<Name>.js), `--media` replaces the demo scenes.
 */

import { createServer } from "node:http";
import { cpSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join, relative, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { fileUnder, kAssetDirs, kSiteDir, sendFile } from "../lib/static.mjs";
import { isPassName, kebabCase, passFiles } from "../lib/scaffold.mjs";

const argv = process.argv.slice(2);
const option = (name, fallback) => {
    const i = argv.indexOf(name);
    return i >= 0 ? argv[i + 1] : fallback;
};
if (argv.includes("-h") || argv.includes("--help")) {
    console.log("usage: web-falcor [--port 5173] [--plugins <dir>] [--media <dir>] [--site <dir>]\n       web-falcor new <Name>\n       web-falcor dev [--port 5173]\n       web-falcor assets <dir>\n       web-falcor py2js <file-or-dir>... [--out <dir>] [--root <dir>] [--check]");
    process.exit(0);
}
if (argv[0] === "py2js") {
    // The converter (transpiled PyToJs.ts) and the viewer's own Pyodide parse and convert.
    const args = (await import("../lib/py2js.mjs")).parseArgs(argv.slice(1));
    if (!args.inputs.length) {
        console.error("usage: web-falcor py2js <file-or-dir>... [--out <dir>] [--root <dir>] [--check]");
        process.exit(1);
    }
    const core = await import("../lib/pytojs-core.mjs");
    const { convertFiles } = await import("../lib/py2js.mjs");
    const warnings = await convertFiles({ ...args, core, pyodideDir: join(kSiteDir, "node_modules/pyodide") });
    process.exit(args.check && warnings ? 1 : 0);
}
if (argv[0] === "assets") {
    if (!argv[1]) {
        console.error("usage: web-falcor assets <dir>");
        process.exit(1);
    }
    for (const d of kAssetDirs) cpSync(join(kSiteDir, d), join(resolve(argv[1]), d), { recursive: true });
    console.log(`runtime assets copied to ${resolve(argv[1])}; serve it and call setAssetBase("<its URL>/")`);
    process.exit(0);
}

const port = Number(option("--port", "5173"));

if (argv[0] === "new") {
    const name = argv[1];
    if (!name || !isPassName(name)) {
        console.error("usage: web-falcor new <Name>   (<Name>: PascalCase pass type, e.g. MyBlur)");
        process.exit(1);
    }
    const dir = resolve(kebabCase(name));
    if (existsSync(dir)) {
        console.error(`${dir} already exists`);
        process.exit(1);
    }
    mkdirSync(dir, { recursive: true });
    for (const [file, text] of passFiles(name, "standalone")) writeFileSync(join(dir, file), text);
    console.log(`Created the ${name} plugin in ${dir}\n\nNext:\n  cd ${relative(process.cwd(), dir)}\n  npm install\n  npm run dev`);
    process.exit(0);
}

if (argv[0] === "dev") {
    // Vite comes from the plugin project (a devDependency there), with its vite.config.js.
    const root = process.cwd();
    const pkg = existsSync(join(root, "package.json")) ? JSON.parse(readFileSync(join(root, "package.json"), "utf8")) : {};
    let vite;
    try {
        // Vite's ESM entry (resolving "vite" through require would load its deprecated CJS build).
        const vitePkg = createRequire(join(root, "package.json")).resolve("vite/package.json");
        const entry = JSON.parse(readFileSync(vitePkg, "utf8")).exports["."].import;
        vite = await import(pathToFileURL(join(dirname(vitePkg), typeof entry === "string" ? entry : entry.default)).href);
    } catch {
        console.error("web-falcor dev needs vite in this project: npm install -D vite");
        process.exit(1);
    }
    const { webFalcorViewer } = await import("../lib/vite.mjs");
    const server = await vite.createServer({ root, plugins: [webFalcorViewer()], server: { port } });
    await server.listen();
    // package.json "webFalcor": the scene (under /Falcor/media/), plugin module and graph script to open.
    const { scene, plugin, script } = pkg.webFalcor ?? {};
    const query = [scene && `scene=${scene}`, plugin && `plugin=/${plugin}`, script && `script=/${script}`].filter(Boolean).join("&");
    console.log(`web-falcor viewer with ${pkg.name ?? "this project"}: http://localhost:${server.config.server.port}/${query ? `?${query}` : ""}`);
    await new Promise(() => {});
}

const site = resolve(option("--site", kSiteDir));
const mounts = [
    ["/plugins/", option("--plugins")],
    ["/Falcor/media/", option("--media")],
]
    .filter(([, dir]) => dir)
    .map(([prefix, dir]) => [prefix, resolve(dir)]);

if (!existsSync(join(site, "index.html"))) {
    console.error(`no viewer build at ${site} (in the repo: npm run build:web)`);
    process.exit(1);
}

createServer((req, res) => {
    const path = new URL(req.url ?? "/", "http://localhost").pathname;
    const mount = mounts.find(([prefix]) => path.startsWith(prefix));
    const file = mount ? fileUnder(mount[1], path.slice(mount[0].length)) : fileUnder(site, path.slice(1));
    if (file) sendFile(req, res, file);
    else res.writeHead(404, { "content-type": "text/plain" }).end("not found");
}).listen(port, () => {
    console.log(`web-falcor viewer: http://localhost:${port}/`);
    for (const [prefix, dir] of mounts) console.log(`  ${prefix} -> ${dir}`);
});
