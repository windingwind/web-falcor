#!/usr/bin/env node
/**
 * Serves the prebuilt Mogwai viewer (`npx @web-falcor/mogwai`), plus local plugins and scenes.
 *
 *   web-falcor [--port 5173] [--plugins <dir>] [--media <dir>] [--site <dir>]
 *   web-falcor assets <dir>      copy the runtime assets for @web-falcor/falcor apps into <dir>
 *
 * `--plugins` is served at /plugins/ (load one with ?plugin=/plugins/<Name>/<Name>.js), `--media` replaces the demo scenes.
 */

import { createServer } from "node:http";
import { cpSync, existsSync } from "node:fs";
import { join, resolve } from "node:path";
import { fileUnder, kAssetDirs, kSiteDir, sendFile } from "../lib/static.mjs";

const argv = process.argv.slice(2);
const option = (name, fallback) => {
    const i = argv.indexOf(name);
    return i >= 0 ? argv[i + 1] : fallback;
};
if (argv.includes("-h") || argv.includes("--help")) {
    console.log("usage: web-falcor [--port 5173] [--plugins <dir>] [--media <dir>] [--site <dir>]\n       web-falcor assets <dir>");
    process.exit(0);
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
