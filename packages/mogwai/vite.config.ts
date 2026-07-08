import { defineConfig } from "vite";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { realpathSync } from "node:fs";

// This package's Vite root is packages/mogwai, but two asset trees the viewer
// fetches at runtime live outside it:
//   • pyodide  — a repo-root dependency npm workspaces hoist to <repo>/node_modules
//   • /Falcor  — Falcor's source + media (media is a symlink into the packman
//                cache), sitting at the repo root
// A browser request for either misses the Vite root and gets the index.html SPA
// fallback (200 text/html), which then blows up as a bad JS module / a .pyscene
// that is really HTML. Resolve their real locations and redirect the requests to
// them through Vite's /@fs/ endpoint (permitted by the server.fs.allow entries).
const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "../..");
const pyodideDir = dirname(createRequire(import.meta.url).resolve("pyodide/package.json"));
const falcorDir = resolve(repoRoot, "Falcor");

// Falcor/media is a symlink into the packman cache (outside the repo), so the
// /@fs/ read for a scene escapes server.fs.allow unless we whitelist the real
// target. Resolve it dynamically (machine-specific path, never hard-coded).
let falcorMediaReal: string | undefined;
try {
    falcorMediaReal = realpathSync(resolve(falcorDir, "media"));
} catch {
    falcorMediaReal = undefined; // media not downloaded yet — run `npm run download:scenes`
}

const rewrites = [
    { prefix: "/node_modules/pyodide/", target: pyodideDir },
    { prefix: "/Falcor/", target: falcorDir },
    // Shader sources (packages/falcor/shaders/**, the generated manifest) and the
    // Slang wasm compiler live at the repo root too — the program system fetches
    // them at runtime, mirroring the GPU test harness (which roots Vite at the repo).
    { prefix: "/packages/", target: resolve(repoRoot, "packages") },
    { prefix: "/tools/", target: resolve(repoRoot, "tools") },
];

export default defineConfig({
    server: {
        port: 5173,
        fs: {
            // Serve workspace packages, generated shader artifacts, and the
            // symlinked Falcor media bundle.
            allow: ["../..", ...(falcorMediaReal ? [falcorMediaReal] : [])],
        },
    },
    plugins: [
        {
            name: "web-falcor:serve-external-assets",
            configureServer(server) {
                server.middlewares.use((req, _res, next) => {
                    for (const { prefix, target } of rewrites) {
                        if (req.url?.startsWith(prefix)) {
                            req.url = `/@fs${target}/${req.url.slice(prefix.length)}`;
                            break;
                        }
                    }
                    next();
                });
            },
        },
    ],
});
