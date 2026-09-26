/**
 * Importer plugins, mirroring native's Importer plugin classes: a module registers a parser for file extensions
 * (through the `@web-falcor/falcor` export or the `globalThis.webFalcorPlugins` hook, see loadPluginLibrary) and
 * SceneBuilder.importScene uses it ahead of the built-in importers. A plugin returns what GltfImporter.parseToDescs
 * does: meshes, materials, the node graph, animations, lights and an optional camera.
 */

import type { GltfImporter } from "./GltfImporter.js";
import type { TextureManager } from "../Material/TextureManager.js";

export type ImporterResult = Awaited<ReturnType<typeof GltfImporter.parseToDescs>>;
export type ImporterPlugin = (bytes: Uint8Array, url: string, textureManager: TextureManager, options: { assumeLinearSpaceTextures?: boolean; useOriginalTangentSpace?: boolean }) => Promise<ImporterResult>;

const importers = new Map<string, ImporterPlugin>();

/** Registers `importer` for file extensions (without the dot, case-insensitive); a later registration wins. */
export function registerImporter(extensions: string[], importer: ImporterPlugin): void {
    for (const ext of extensions) importers.set(ext.toLowerCase().replace(/^\./, ""), importer);
}

/** The plugin importer for `path`'s extension, if one is registered. */
export function getRegisteredImporter(path: string): ImporterPlugin | undefined {
    const clean = path.split(/[?#]/)[0]!;
    return importers.get(clean.slice(clean.lastIndexOf(".") + 1).toLowerCase());
}

export function getRegisteredImporterExtensions(): string[] {
    return [...importers.keys()];
}
