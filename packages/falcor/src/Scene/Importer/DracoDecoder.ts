/**
 * Draco mesh decoding for `KHR_draco_mesh_compression`.
 *
 * Wraps Google's official decoder (the `draco3d` package's emscripten build,
 * loaded the same way `assimpjs` is). Native Falcor gets Draco support through
 * assimp; the web importer is its own TS reader, so it drives the decoder here.
 */

import { RuntimeError } from "../../Core/Error.js";

/** Minimal shape of the emscripten decoder module we use. */
interface DracoModule {
    Decoder: new () => DracoDecoderApi;
    DecoderBuffer: new () => { Init(data: Uint8Array, length: number): void; __destroy__?(): void };
    Mesh: new () => DracoMesh;
    TRIANGULAR_MESH: number;
    DT_FLOAT32: number;
    DT_UINT32: number;
    _malloc(bytes: number): number;
    _free(ptr: number): void;
    HEAPU8: Uint8Array;
    destroy(object: unknown): void;
}

interface DracoMesh {
    num_faces(): number;
    num_points(): number;
}

interface DracoAttribute {
    num_components(): number;
}

interface DracoDecoderApi {
    GetEncodedGeometryType(buffer: unknown): number;
    DecodeBufferToMesh(buffer: unknown, mesh: DracoMesh): { ok(): boolean; error_msg(): string };
    GetAttributeByUniqueId(mesh: DracoMesh, uniqueId: number): DracoAttribute;
    GetAttributeDataArrayForAllPoints(mesh: DracoMesh, attribute: DracoAttribute, dataType: number, byteLength: number, ptr: number): boolean;
    GetTrianglesUInt32Array(mesh: DracoMesh, byteLength: number, ptr: number): boolean;
}

let modulePromise: Promise<DracoModule> | null = null;

/** Loads the decoder (emscripten UMD script from node_modules), once per page. */
export function getDracoDecoderModule(): Promise<DracoModule> {
    if (modulePromise) return modulePromise;
    modulePromise = (async () => {
        const g = globalThis as { DracoDecoderModule?: (opts?: object) => Promise<DracoModule> };
        if (!g.DracoDecoderModule) {
            await new Promise<void>((resolve, reject) => {
                const script = document.createElement("script");
                script.src = "/node_modules/draco3d/draco_decoder_nodejs.js";
                script.onload = () => resolve();
                script.onerror = () => reject(new RuntimeError("DracoDecoder: failed to load draco3d"));
                document.head.appendChild(script);
            });
        }
        if (!g.DracoDecoderModule) throw new RuntimeError("DracoDecoder: draco3d did not register its module factory");
        return g.DracoDecoderModule({ locateFile: (file: string) => `/node_modules/draco3d/${file}` });
    })();
    return modulePromise;
}

export interface DecodedDracoMesh {
    /** Vertex attributes by glTF semantic (POSITION, NORMAL, TEXCOORD_0, …), as floats. */
    attributes: Map<string, Float32Array>;
    indices: Uint32Array;
    vertexCount: number;
}

/**
 * Decodes one Draco-compressed primitive.
 *
 * @param data The `KHR_draco_mesh_compression` buffer view's bytes.
 * @param attributeIds glTF semantic -> Draco attribute unique id, from the extension.
 */
export async function decodeDracoMesh(data: Uint8Array, attributeIds: Record<string, number>): Promise<DecodedDracoMesh> {
    const draco = await getDracoDecoderModule();
    const decoder = new draco.Decoder();
    const buffer = new draco.DecoderBuffer();
    const mesh = new draco.Mesh();
    try {
        buffer.Init(data, data.length);
        if (decoder.GetEncodedGeometryType(buffer) !== draco.TRIANGULAR_MESH) {
            throw new RuntimeError("DracoDecoder: only triangular meshes are supported");
        }
        const status = decoder.DecodeBufferToMesh(buffer, mesh);
        if (!status.ok()) throw new RuntimeError(`DracoDecoder: ${status.error_msg()}`);

        const vertexCount = mesh.num_points();
        /** Copies `byteLength` bytes out of the wasm heap through a scratch allocation. */
        const readFromHeap = <T>(byteLength: number, fill: (ptr: number) => boolean, make: (bytes: Uint8Array) => T): T => {
            const ptr = draco._malloc(byteLength);
            try {
                if (!fill(ptr)) throw new RuntimeError("DracoDecoder: failed to read decoded data");
                // Copy before freeing — and before any later allocation can move the heap.
                return make(draco.HEAPU8.slice(ptr, ptr + byteLength));
            } finally {
                draco._free(ptr);
            }
        };

        const indexCount = mesh.num_faces() * 3;
        const indices = readFromHeap(
            indexCount * 4,
            (ptr) => decoder.GetTrianglesUInt32Array(mesh, indexCount * 4, ptr),
            (bytes) => new Uint32Array(bytes.buffer, bytes.byteOffset, indexCount),
        );

        const attributes = new Map<string, Float32Array>();
        for (const [semantic, uniqueId] of Object.entries(attributeIds)) {
            const attribute = decoder.GetAttributeByUniqueId(mesh, uniqueId);
            const components = attribute.num_components();
            const valueCount = vertexCount * components;
            // Float output covers every glTF attribute the importer consumes; Draco
            // dequantizes its internal representation on the way out.
            attributes.set(
                semantic,
                readFromHeap(
                    valueCount * 4,
                    (ptr) => decoder.GetAttributeDataArrayForAllPoints(mesh, attribute, draco.DT_FLOAT32, valueCount * 4, ptr),
                    (bytes) => new Float32Array(bytes.buffer, bytes.byteOffset, valueCount),
                ),
            );
        }
        return { attributes, indices, vertexCount };
    } finally {
        draco.destroy(mesh);
        draco.destroy(buffer);
        draco.destroy(decoder);
    }
}
