/**
 * Local media: files the user picked or dropped (a folder, or several files) at run time, served under a
 * virtual same-origin path so the importers load them by URL like served media.
 *
 * Served in the page, not by a Service Worker: Python's open() and glob read files with synchronous XHR, which
 * would deadlock on a worker that asks the page for the bytes. So fetch() is wrapped while anything is
 * mounted, and the synchronous readers call readLocalFileSync / listLocalDirectory directly.
 */

import { Logger } from "../Utils/Logger.js";

/** Path segment that marks a mounted file's URL; a URL is local iff its path contains it. */
const kMarker = "/__webfalcor_local/";
/** Files Python may open() synchronously are read into memory at mount time (up to this size). */
const kSyncExtensions = new Set(["py", "pyscene", "js", "json", "txt", "csv", "ini", "cfg", "yaml", "yml", "xml"]);
const kSyncMaxBytes = 16 << 20;

interface LocalEntry {
    file: File;
    /** Bytes for synchronous readers, if the file was preloaded. */
    bytes?: Uint8Array;
}

/** Mounted files by URL path (`/__webfalcor_local/<id>/<relative path>`). */
const entries = new Map<string, LocalEntry>();
let mountCount = 0;
let originalFetch: typeof fetch | null = null;

export interface LocalMount {
    /** URL of the mount's root directory, with a trailing slash. */
    baseUrl: string;
    /** Every mounted file's URL path, in input order. */
    urls: string[];
}

/**
 * Mounts `files` (relative path → File) under a fresh virtual directory and returns its URLs. Relative paths
 * keep their directories, so a scene's references to sibling files resolve as they do on disk.
 */
export async function mountLocalFiles(files: Iterable<readonly [string, File]>): Promise<LocalMount> {
    const baseUrl = `${kMarker}${(Date.now() % 1e8).toString(36)}${(mountCount++).toString(36)}/`;
    const urls: string[] = [];
    const preloads: Promise<void>[] = [];
    for (const [relPath, file] of files) {
        const url = baseUrl + relPath.replace(/^\/+/, "");
        const entry: LocalEntry = { file };
        entries.set(url, entry);
        urls.push(url);
        if (kSyncExtensions.has(extensionOf(url)) && file.size <= kSyncMaxBytes) {
            preloads.push(file.arrayBuffer().then((b) => void (entry.bytes = new Uint8Array(b))));
        }
    }
    await Promise.all(preloads);
    installFetchHook();
    return { baseUrl, urls };
}

/** Forgets every file under `baseUrl` (a LocalMount's). */
export function unmountLocalFiles(baseUrl: string): void {
    for (const url of [...entries.keys()]) if (url.startsWith(baseUrl)) entries.delete(url);
}

/** The mounted path of `url` (a URL or path), or null if it isn't local media. */
export function localMediaPath(url: string | URL): string | null {
    let path: string;
    try {
        const u = new URL(url, globalThis.location?.href ?? "http://localhost/");
        if (globalThis.location && u.origin !== globalThis.location.origin) return null;
        path = decodeURIComponent(u.pathname);
    } catch {
        return null;
    }
    // Tolerates an asset-base prefix (assetUrl() applied to a local path).
    const at = path.indexOf(kMarker);
    return at < 0 ? null : path.slice(at);
}

export function isLocalMediaUrl(url: string | URL): boolean {
    return localMediaPath(url) !== null;
}

/** The mounted File at `url`, if any. */
export function getLocalFile(url: string | URL): File | undefined {
    const path = localMediaPath(url);
    return path === null ? undefined : entries.get(path)?.file;
}

/**
 * Synchronous read for Python's open(): null if `url` isn't local media, undefined if the file is missing or
 * too large to have been preloaded.
 */
export function readLocalFileSync(url: string | URL): Uint8Array | undefined | null {
    const path = localMediaPath(url);
    if (path === null) return null;
    const entry = entries.get(path);
    if (entry && !entry.bytes) Logger.warning(`LocalMedia: '${path}' can't be read synchronously (only small text files are preloaded)`);
    return entry?.bytes;
}

/** A mounted directory's files and subdirectories (names), or null if `url` isn't local media. */
export function listLocalDirectory(url: string | URL): { files: string[]; dirs: string[] } | null {
    const path = localMediaPath(url);
    if (path === null) return null;
    const dir = path.endsWith("/") ? path : `${path}/`;
    const files = new Set<string>();
    const dirs = new Set<string>();
    for (const key of entries.keys()) {
        if (!key.startsWith(dir)) continue;
        const rest = key.slice(dir.length);
        const slash = rest.indexOf("/");
        if (slash < 0) files.add(rest);
        else dirs.add(rest.slice(0, slash));
    }
    return { files: [...files], dirs: [...dirs] };
}

/** Routes fetch() of local media to the mounted files; everything else goes to the original fetch. */
function installFetchHook(): void {
    if (originalFetch) return;
    const base = globalThis.fetch.bind(globalThis);
    originalFetch = base;
    globalThis.fetch = (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
        const url = input instanceof Request ? input.url : input;
        const path = localMediaPath(url);
        if (path === null) return base(input, init);
        const entry = entries.get(path);
        if (!entry) return Promise.resolve(new Response(null, { status: 404, statusText: "Not Found" }));
        const method = (init?.method ?? (input instanceof Request ? input.method : "GET")).toUpperCase();
        const headers = { "content-type": contentTypeOf(path), "content-length": String(entry.file.size) };
        return Promise.resolve(new Response(method === "HEAD" ? null : entry.file, { status: 200, headers }));
    };
}

function extensionOf(path: string): string {
    const name = path.slice(path.lastIndexOf("/") + 1);
    const dot = name.lastIndexOf(".");
    return dot < 0 ? "" : name.slice(dot + 1).toLowerCase();
}

function contentTypeOf(path: string): string {
    const ext = extensionOf(path);
    if (ext === "js" || ext === "mjs") return "text/javascript";
    if (ext === "json") return "application/json";
    if (ext === "png" || ext === "jpg" || ext === "jpeg" || ext === "webp") return `image/${ext === "jpg" ? "jpeg" : ext}`;
    if (kSyncExtensions.has(ext)) return "text/plain";
    return "application/octet-stream";
}

/** Files from an `<input type="file" webkitdirectory>` (or `multiple`), keyed by their path in the picked folder. */
export function filesFromInput(list: FileList): [string, File][] {
    return Array.from(list, (f) => [f.webkitRelativePath || f.name, f] as [string, File]);
}

/**
 * Files from a drop, with dropped folders walked recursively. Call it synchronously in the drop handler: the
 * browser invalidates `dataTransfer` once the handler returns.
 */
export function filesFromDataTransfer(dt: DataTransfer): Promise<[string, File][]> {
    const roots = Array.from(dt.items, (item) => item.webkitGetAsEntry?.()).filter((e): e is FileSystemEntry => !!e);
    if (roots.length === 0) return Promise.resolve(Array.from(dt.files, (f) => [f.name, f] as [string, File]));
    return Promise.all(roots.map((e) => walkEntry(e, ""))).then((lists) => lists.flat());
}

/** Whether a drop contains a folder (so the caller can mount it rather than load a single file). */
export function dataTransferHasDirectory(dt: DataTransfer): boolean {
    return Array.from(dt.items).some((item) => item.webkitGetAsEntry?.()?.isDirectory);
}

async function walkEntry(entry: FileSystemEntry, prefix: string): Promise<[string, File][]> {
    const path = prefix + entry.name;
    if (entry.isFile) {
        const file = await new Promise<File>((resolve, reject) => (entry as FileSystemFileEntry).file(resolve, reject));
        return [[path, file]];
    }
    const reader = (entry as FileSystemDirectoryEntry).createReader();
    const children: FileSystemEntry[] = [];
    // readEntries returns batches (100 in Chromium) until an empty one.
    for (;;) {
        const batch = await new Promise<FileSystemEntry[]>((resolve, reject) => reader.readEntries(resolve, reject));
        if (batch.length === 0) break;
        children.push(...batch);
    }
    const lists = await Promise.all(children.map((c) => walkEntry(c, `${path}/`)));
    return lists.flat();
}
