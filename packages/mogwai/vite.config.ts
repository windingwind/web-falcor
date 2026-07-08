import { defineConfig } from "vite";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { realpathSync } from "node:fs";

// Root Vite at the repo (like tests/gpu/harness) so asset trees outside mogwai
// — /Falcor, /packages/falcor/shaders, /tools, /node_modules/pyodide — serve as static files.
const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "../..");
const appEntry = "/packages/mogwai/index.html";

// Falcor/media is a symlink into the packman cache (outside the repo); whitelist
// its real target so Vite will serve scene assets through the symlink.
let falcorMediaReal: string | undefined;
try {
    falcorMediaReal = realpathSync(resolve(repoRoot, "Falcor/media"));
} catch {
    falcorMediaReal = undefined; // media not downloaded yet — run `npm run download:scenes`
}

export default defineConfig({
    root: repoRoot,
    server: {
        port: 5173,
        fs: {
            allow: [repoRoot, ...(falcorMediaReal ? [falcorMediaReal] : [])],
        },
    },
    plugins: [
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
