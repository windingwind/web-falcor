/**
 * Vite plugin for apps built on @web-falcor/falcor: serves the runtime assets at `<base>web-falcor/` in dev, copies
 * them into the build, keeps the packages out of pre-bundling (it breaks their wasm URLs) and targets ES2022.
 * The app then calls `setAssetBase(import.meta.env.BASE_URL + "web-falcor/")`.
 */

import { cpSync } from "node:fs";
import { join, resolve } from "node:path";
import { fileUnder, kAssetDirs, kSiteDir, sendFile } from "./static.mjs";

export function webFalcor({ path = "web-falcor/" } = {}) {
    let outDir = "";
    let assetPath = path;
    return {
        name: "web-falcor",
        // Every WebGPU browser runs ES2022 (top-level await included); an app's own target wins.
        config: (user) => ({ optimizeDeps: { exclude: ["@web-falcor/falcor", "@web-falcor/render-passes"] }, ...(user.build?.target ? {} : { build: { target: "es2022" } }) }),
        configResolved(config) {
            outDir = resolve(config.root, config.build.outDir);
            assetPath = config.base + path;
        },
        configureServer(server) {
            // Served raw, ahead of Vite's transforms: slang-wasm and Pyodide are imported as-is.
            server.middlewares.use((req, res, next) => {
                const url = new URL(req.url ?? "/", "http://localhost").pathname;
                if (!url.startsWith(assetPath)) return next();
                const rel = url.slice(assetPath.length);
                const file = kAssetDirs.includes(rel.split("/")[0]) ? fileUnder(kSiteDir, rel) : null;
                if (file) sendFile(req, res, file);
                else next();
            });
        },
        closeBundle() {
            for (const d of kAssetDirs) cpSync(join(kSiteDir, d), join(outDir, path, d), { recursive: true });
        },
    };
}
