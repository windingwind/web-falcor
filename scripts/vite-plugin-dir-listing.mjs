/**
 * Vite dev-server plugin: GET /__webfalcor/ls?path=<served path> returns {"files": [...], "dirs": [...]} for a
 * directory under the served roots (symlinks such as Falcor/media followed). The web AssetResolver's
 * resolvePathPattern and Python's glob/os.listdir in scene scripts use it; static hosting has no listing (docs §9).
 */

import { readdirSync, realpathSync, statSync } from "node:fs";
import { join, sep } from "node:path";

/** @param {{ root: string, allow?: string[] }} options served root, plus real paths it may link into */
export function dirListing({ root, allow = [] }) {
    const allowed = [root, ...allow].map((p) => realpathSync(p));
    const inside = (p) => allowed.some((a) => p === a || p.startsWith(a + sep));
    return {
        name: "web-falcor:dir-listing",
        configureServer(server) {
            server.middlewares.use("/__webfalcor/ls", (req, res) => {
                const path = new URL(req.url ?? "", "http://x").searchParams.get("path") ?? "";
                let real;
                try {
                    real = realpathSync(join(root, decodeURIComponent(path)));
                } catch {
                    real = undefined;
                }
                if (!real || !inside(real) || !statSync(real).isDirectory()) {
                    res.statusCode = 404;
                    res.end();
                    return;
                }
                const files = [];
                const dirs = [];
                for (const entry of readdirSync(real, { withFileTypes: true })) {
                    let isDir = entry.isDirectory();
                    let isFile = entry.isFile();
                    if (entry.isSymbolicLink()) {
                        try {
                            const st = statSync(join(real, entry.name));
                            [isDir, isFile] = [st.isDirectory(), st.isFile()];
                        } catch {
                            continue;
                        }
                    }
                    if (isDir) dirs.push(entry.name);
                    else if (isFile) files.push(entry.name);
                }
                res.setHeader("Content-Type", "application/json");
                res.end(JSON.stringify({ files: files.sort(), dirs: dirs.sort() }));
            });
        },
    };
}
