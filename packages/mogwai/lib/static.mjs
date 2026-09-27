/** Static serving of the prebuilt viewer and its runtime assets, shared by the CLI and the Vite plugin. */

import { createReadStream, existsSync, statSync } from "node:fs";
import { extname, join, normalize, sep } from "node:path";
import { fileURLToPath } from "node:url";

/** The built site: the viewer app plus the runtime assets (packages/mogwai/dist, or dist/ in the npm package). */
export const kSiteDir = fileURLToPath(new URL("../dist/", import.meta.url));
/** The asset tree setAssetBase() points at: everything but the viewer app itself. */
export const kAssetDirs = ["Falcor", "packages", "tools", "node_modules"];

const kTypes = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".json": "application/json",
    ".wasm": "application/wasm",
    ".css": "text/css",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".txt": "text/plain; charset=utf-8",
    ".md": "text/plain; charset=utf-8",
};

/** The file `rel` names under `root`, or null if it escapes the root or doesn't exist. */
export function fileUnder(root, rel) {
    const base = normalize(root).replace(/[\\/]$/, "");
    let file = normalize(join(base, decodeURIComponent(rel)));
    if (file !== base && !file.startsWith(base + sep)) return null;
    if (existsSync(file) && statSync(file).isDirectory()) file = join(file, "index.html");
    return existsSync(file) ? file : null;
}

/** Streams `file` with its content type. */
export function sendFile(req, res, file) {
    res.writeHead(200, { "content-type": kTypes[extname(file)] ?? "application/octet-stream" });
    if (req.method === "HEAD") res.end();
    else createReadStream(file).pipe(res);
}
