/**
 * Base URL of the runtime's own assets (shaders, slang-wasm, Pyodide, fonts, media). Asset paths keep the
 * dev server's repo layout (`/Falcor/...`, `/tools/...`) and resolve under this base; the dev server's is "/".
 */

let assetBase = "/";

/** Web analog of getProjectDirectory()/"media": the served Falcor media tree (follows setAssetBase). */
export let kProjectMediaUrl = "/Falcor/media";
/** Where scripts/setup-web.mjs puts the pinned Pyodide packages (follows setAssetBase). */
export let kPyodidePackagesUrl = "/tools/pyodide-packages/";
/** Default font location, mirrors getRuntimeDirectory()/data/framework/fonts (follows setAssetBase). */
export let kDefaultFontUrl = "/Falcor/data/framework/fonts/dejavu-sans-mono-14";

/** Serves the assets from `base` (e.g. "/web-falcor/" or a CDN URL); call before loading anything. */
export function setAssetBase(base: string): void {
    assetBase = base.endsWith("/") ? base : `${base}/`;
    kProjectMediaUrl = assetUrl("/Falcor/media");
    kPyodidePackagesUrl = assetUrl("/tools/pyodide-packages/");
    kDefaultFontUrl = assetUrl("/Falcor/data/framework/fonts/dejavu-sans-mono-14");
}

export function getAssetBase(): string {
    return assetBase;
}

/** Maps a root-relative asset path ("/tools/slang-wasm/slang-wasm.js") under the asset base; other URLs pass through. */
export function assetUrl(path: string): string {
    return path.startsWith("/") && !path.startsWith("//") ? assetBase + path.slice(1) : path;
}
