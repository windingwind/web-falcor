/**
 * Importer plugins, mirroring native's Importer plugin classes: a module registers a parser for file extensions
 * (through the `@web-falcor/falcor` export or the `globalThis.webFalcorPlugins` hook, see loadPluginLibrary) and
 * SceneBuilder.importScene uses it ahead of the built-in importers. A plugin returns what GltfImporter.parseToDescs
 * does: meshes, materials, the node graph, animations, lights and an optional camera.
 */

import type { GltfImporter } from "./GltfImporter.js";
import type { TextureManager } from "../Material/TextureManager.js";
import { PluginManager } from "../../Core/Plugin.js";

export type ImporterResult = Awaited<ReturnType<typeof GltfImporter.parseToDescs>>;
export type ImporterPlugin = (bytes: Uint8Array, url: string, textureManager: TextureManager, options: { assumeLinearSpaceTextures?: boolean; useOriginalTangentSpace?: boolean }) => Promise<ImporterResult>;

/** The Importer plugin base class's key and info (native Importer::PluginInfo: desc, extensions). */
export const Importer = { name: "Importer" };
export interface ImporterInfo {
    desc: string;
    extensions: string[];
}

/**
 * Registers `importer` for file extensions (without the dot, case-insensitive) as an Importer class in the
 * PluginManager (type defaults to e.g. "TriImporter"); for an extension, the latest registration wins.
 */
export function registerImporter(extensions: string[], importer: ImporterPlugin, type?: string, desc = ""): void {
    const exts = extensions.map((e) => e.toLowerCase().replace(/^\./, ""));
    const name = type ?? `${exts[0]!.charAt(0).toUpperCase()}${exts[0]!.slice(1)}Importer`;
    // create() returns the importer, as native's create returns an Importer instance.
    PluginManager.instance().registerClass(Importer, name, { desc, extensions: exts }, () => importer);
    order = [name, ...order.filter((n) => n !== name)];
}
let order: string[] = [];

/** The plugin importer for `path`'s extension, if one is registered. */
export function getRegisteredImporter(path: string): ImporterPlugin | undefined {
    const clean = path.split(/[?#]/)[0]!;
    const ext = clean.slice(clean.lastIndexOf(".") + 1).toLowerCase();
    const infos = new Map(PluginManager.instance().getInfos<ImporterInfo>(Importer));
    const type = order.find((n) => infos.get(n)?.extensions.includes(ext));
    return type ? (PluginManager.instance().createClass<() => ImporterPlugin>(Importer, type) ?? undefined) : undefined;
}

export function getRegisteredImporterExtensions(): string[] {
    return [...new Set(PluginManager.instance().getInfos<ImporterInfo>(Importer).flatMap(([, info]) => info.extensions))];
}
