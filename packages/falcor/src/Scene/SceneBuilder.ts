/**
 * SceneBuilder bridge for .pyscene execution (user decision DESIGN.md §11.1:
 * the .py path is primary). Mirrors the subset of Falcor's SceneBuilder python
 * API used by scene files: importScene, TriangleMesh factories, BasicMaterial
 * subclasses, addNode/addMeshInstance, analytic lights and env maps.
 *
 * Python execution is synchronous, but asset loading on the web is not — so
 * builder calls record commands, and resolve() fetches assets and constructs
 * the Scene afterwards.
 */

import type { Device } from "../Core/API/Device.js";
import { Scene, type SceneMaterialDesc, type SceneMeshDesc } from "./Scene.js";
import { GltfImporter } from "./Importer/GltfImporter.js";
import { TextureManager } from "./Material/TextureManager.js";
import { EnvMap } from "./Lights/EnvMap.js";
import { generateTangents } from "./TangentSpace.js";
import { LightType, type AnalyticLight, type StaticVertex } from "./SceneData.js";
import { MaterialType } from "./Material/MaterialData.js";
import { float2, float3, float4 } from "../Utils/Math/Vector.js";
import { float4x4, matrixFromTranslation, matrixFromScaling, mulMat } from "../Utils/Math/Matrix.js";
import { RuntimeError } from "../Core/Error.js";

/** TriangleMesh geometry in local space (mirrors Falcor::TriangleMesh). */
export interface TriangleMeshDesc {
    vertices: StaticVertex[];
    indices: Uint32Array;
}

/** Mirrors TriangleMesh::createQuad (XZ plane, +Y normal). */
export const TriangleMesh = {
    createQuad(size: float2 = new float2(1, 1)): TriangleMeshDesc {
        const hx = 0.5 * size.x;
        const hy = 0.5 * size.y;
        const n = new float3(0, 1, 0);
        const t0 = new float4(0, 0, 0, 0);
        const vertices: StaticVertex[] = [
            { position: new float3(-hx, 0, -hy), normal: n, tangent: t0, texCrd: new float2(0, 0) },
            { position: new float3(hx, 0, -hy), normal: n, tangent: t0, texCrd: new float2(1, 0) },
            { position: new float3(-hx, 0, hy), normal: n, tangent: t0, texCrd: new float2(0, 1) },
            { position: new float3(hx, 0, hy), normal: n, tangent: t0, texCrd: new float2(1, 1) },
        ];
        return { vertices, indices: new Uint32Array([2, 1, 0, 1, 2, 3]) };
    },
};

/** Material bridge mirroring the BasicMaterial python properties. */
export class MaterialBridge {
    baseColor = new float4(1, 1, 1, 1);
    /** specularParams (occlusion/roughness/metallic; BasicMaterialData default is all-zero). */
    specularParams = new float4(0, 0, 0, 0);
    transmissionColor = new float3(1, 1, 1);
    emissiveColor = new float3(0, 0, 0);
    emissiveFactor = 1;
    doubleSided = false;

    constructor(
        public readonly materialType: MaterialType,
        public readonly name: string,
    ) {}

    /** ClothMaterial/BasicMaterial::setRoughness -> specular.g. */
    set roughness(r: number | float2) {
        if (typeof r === "number") {
            this.specularParams = new float4(this.specularParams.x, r, this.specularParams.z, this.specularParams.w);
        } else {
            // PBRTConductorMaterial::setRoughness(float2) -> specular.rg.
            this.specularParams = new float4(r.x, r.y, this.specularParams.z, this.specularParams.w);
        }
    }

    toDesc(): SceneMaterialDesc {
        const emissive = this.emissiveColor.x !== 0 || this.emissiveColor.y !== 0 || this.emissiveColor.z !== 0;
        return {
            header: { materialType: this.materialType, doubleSided: this.doubleSided, emissive },
            basic: {
                baseColor: this.baseColor,
                specular: this.specularParams,
                transmission: this.transmissionColor,
                emissive: this.emissiveColor,
                emissiveFactor: this.emissiveFactor,
            },
        };
    }
}

export class LightBridge {
    position = new float3(0, 0, 0);
    intensity = new float3(1, 1, 1);
    direction = new float3(0, -1, 0);
    constructor(
        public readonly lightType: LightType,
        public readonly name: string,
    ) {}
}

/** Transform bridge (composition order Translate * Rotate * Scale, as native). */
export function makeTransform(
    translation: float3 | null,
    rotationEuler: float3 | null,
    rotationEulerDeg: float3 | null,
    scaling: float3 | null,
): float4x4 {
    const rot = rotationEuler ?? (rotationEulerDeg ? new float3((rotationEulerDeg.x * Math.PI) / 180, (rotationEulerDeg.y * Math.PI) / 180, (rotationEulerDeg.z * Math.PI) / 180) : null);
    let m = float4x4.identity();
    if (scaling) m = mulMat(matrixFromScaling(scaling), m);
    if (rot) {
        // R = Rz * Ry * Rx — numerically verified against math::quatFromEulerAngles
        // (Falcor's euler quat expands to the ZYX matrix product).
        const [cx, sx, cy, sy, cz, sz] = [Math.cos(rot.x), Math.sin(rot.x), Math.cos(rot.y), Math.sin(rot.y), Math.cos(rot.z), Math.sin(rot.z)];
        const r = float4x4.identity();
        r.set(0, 0, cy * cz);
        r.set(0, 1, sx * sy * cz - cx * sz);
        r.set(0, 2, cx * sy * cz + sx * sz);
        r.set(1, 0, cy * sz);
        r.set(1, 1, sx * sy * sz + cx * cz);
        r.set(1, 2, cx * sy * sz - sx * cz);
        r.set(2, 0, -sy);
        r.set(2, 1, sx * cy);
        r.set(2, 2, cx * cy);
        m = mulMat(r, m);
    }
    if (translation) m = mulMat(matrixFromTranslation(translation), m);
    return m;
}

interface EnvMapRef {
    path: string;
}

type Command =
    | { kind: "import"; path: string }
    | { kind: "mesh"; mesh: TriangleMeshDesc; material: MaterialBridge; nodeTransform: float4x4 };

export class SceneBuilderBridge {
    private commands: Command[] = [];
    private meshMaterials: MaterialBridge[] = []; // by meshID
    private meshGeometry: TriangleMeshDesc[] = [];
    private meshInstanced = new Map<number, float4x4>();
    private nodes: float4x4[] = [];
    private lights: LightBridge[] = [];
    envMap: EnvMapRef | null = null;

    importScene(path: string): void {
        this.commands.push({ kind: "import", path });
    }

    addTriangleMesh(mesh: TriangleMeshDesc, material: MaterialBridge): number {
        this.meshGeometry.push(mesh);
        this.meshMaterials.push(material);
        return this.meshGeometry.length - 1;
    }

    addNode(_name: string, transform: float4x4): number {
        this.nodes.push(transform);
        return this.nodes.length - 1;
    }

    addMeshInstance(nodeID: number, meshID: number): void {
        const transform = this.nodes[nodeID];
        if (!transform) throw new RuntimeError(`addMeshInstance: unknown node ${nodeID}`);
        this.meshInstanced.set(meshID, transform);
    }

    addLight(light: LightBridge): void {
        this.lights.push(light);
    }

    /** Fetches referenced assets and constructs the Scene. */
    async resolve(device: Device, baseUrl: string): Promise<Scene> {
        const textureManager = new TextureManager();
        const meshes: SceneMeshDesc[] = [];
        const materials: SceneMaterialDesc[] = [];

        for (const cmd of this.commands) {
            if (cmd.kind === "import") {
                const url = baseUrl ? `${baseUrl}/${cmd.path}` : cmd.path;
                const res = await fetch(url);
                if (!res.ok) throw new RuntimeError(`SceneBuilder: failed to fetch '${url}' (${res.status})`);
                const parsed = await GltfImporter.parseToDescs(new Uint8Array(await res.arrayBuffer()), url, textureManager);
                const materialOffset = materials.length;
                materials.push(...parsed.materials);
                for (const m of parsed.meshes) meshes.push({ ...m, materialID: m.materialID + materialOffset });
            }
        }

        // Builder-added meshes (instanced via nodes).
        const materialIDs = new Map<MaterialBridge, number>();
        this.meshGeometry.forEach((geo, meshID) => {
            const transform = this.meshInstanced.get(meshID);
            if (!transform) return; // mesh never instanced
            const mat = this.meshMaterials[meshID]!;
            let materialID = materialIDs.get(mat);
            if (materialID === undefined) {
                materialID = materials.length;
                materials.push(mat.toDesc());
                materialIDs.set(mat, materialID);
            }
            // Local-space tangents when the asset provides none (native MikkTSpace).
            const vertices = geo.vertices.map((v) => ({ ...v }));
            generateTangents(vertices, geo.indices);
            meshes.push({ vertices, indices: geo.indices, materialID, transform });
        });

        const lights: AnalyticLight[] = this.lights.map((l) => ({
            type: l.lightType,
            posW: l.position,
            dirW: l.direction,
            intensity: l.intensity,
        }));

        const scene = new Scene(device, meshes, materials, lights, textureManager);
        if (this.envMap) {
            const url = baseUrl ? `${baseUrl}/${this.envMap.path}` : this.envMap.path;
            scene.setEnvMap(await EnvMap.createFromUrl(device, url));
        }
        return scene;
    }
}
