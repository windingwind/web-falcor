import { defineConfig } from "vite";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { realpathSync } from "node:fs";
import { dirListing } from "../../scripts/vite-plugin-dir-listing.mjs";

// Root Vite at the repo (like tests/gpu/harness) so asset trees outside mogwai
// — /Falcor, /packages/falcor/shaders, /tools, /node_modules/pyodide — serve as static files.
const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "../..");
const appEntry = "/packages/mogwai/index.html";
// Bare specifiers prebuilt plugins import, served as dist/plugin-api/<name>.js.
const pluginApi: Record<string, string> = {
    "web-falcor": resolve(repoRoot, "packages/falcor/src/index.ts"),
    "web-falcor-render-passes": resolve(repoRoot, "packages/render-passes/src/index.ts"),
};
const pluginApiImports = { "@web-falcor/falcor": "web-falcor", "@web-falcor/render-passes": "web-falcor-render-passes" };

// Falcor/media is a symlink into the packman cache (outside the repo); whitelist
// its real target so Vite will serve scene assets through the symlink.
let falcorMediaReal: string | undefined;
try {
    falcorMediaReal = realpathSync(resolve(repoRoot, "Falcor/media"));
} catch {
    falcorMediaReal = undefined; // media not downloaded yet — run `npm run download:scenes`
}

let viteBase = "/";

export default defineConfig({
    root: repoRoot,
    // The app entry lives under the repo root; `vite build` writes packages/mogwai/dist.
    // The packages are entries of their own with stable names, so prebuilt plugins import them via the import map.
    build: {
        rollupOptions: {
            input: { main: resolve(here, "index.html"), ...pluginApi },
            preserveEntrySignatures: "strict",
            output: { entryFileNames: (chunk) => (chunk.name in pluginApi ? `plugin-api/${chunk.name}.js` : "assets/[name]-[hash].js") },
        },
        outDir: resolve(here, "dist"),
        emptyOutDir: true,
    },
    server: {
        port: 5173,
        // Build output (build:web, build:packages, plugin builds) is rewritten wholesale; watching it crashes the server.
        watch: { ignored: ["**/out/**", "**/dist/**"] },
        fs: {
            allow: [repoRoot, ...(falcorMediaReal ? [falcorMediaReal] : [])],
        },
    },
    plugins: [
        dirListing({ root: repoRoot, allow: falcorMediaReal ? [falcorMediaReal] : [] }),
        {
            // Import map for prebuilt plugins (the dev server resolves the bare specifiers itself).
            name: "web-falcor:plugin-import-map",
            apply: "build",
            transformIndexHtml: {
                order: "pre",
                handler: (_html, ctx) => {
                    const base = ctx.server?.config.base ?? viteBase;
                    const imports = Object.fromEntries(Object.entries(pluginApiImports).map(([spec, name]) => [spec, `${base}plugin-api/${name}.js`]));
                    return [{ tag: "script", attrs: { type: "importmap" }, children: JSON.stringify({ imports }), injectTo: "head-prepend" }];
                },
            },
            configResolved(config) {
                viteBase = config.base;
            },
        },
        {
            name: "web-falcor:mogwai-entry",
            configureServer(server) {
                // Open the viewer at / (and /?scene=…) by redirecting to its real
                // path under the repo root, preserving the query string.
                server.middlewares.use((req, res, next) => {
                    if (req.url === "/" || req.url?.startsWith("/?")) {
                        res.statusCode = 302;
                        res.setHeader("Location", appEntry + req.url.slice(1));
                        res.end();
                        return;
                    }
                    next();
                });
            },
        },
    ],
});
