/**
 * LightBVH emissive sampler host mirroring Rendering/Lights/LightBVH.{h,cpp} +
 * LightBVHSampler.{h,cpp}: builds the BVH on the CPU (LightBVHBuilder.ts),
 * uploads nodes/triangleIndices/triangleBitmasks, and provides the sampler
 * defines + bindings (_lightBVH member of EmissiveLightSampler when
 * _EMISSIVE_LIGHT_SAMPLER_TYPE == LIGHT_BVH).
 */

import type { Device } from "../../Core/API/Device.js";
import { Buffer } from "../../Core/API/Buffer.js";
import { MemoryType, ResourceBindFlags } from "../../Core/API/Types.js";
import { DefineList } from "../../Core/Program/DefineList.js";
import type { ShaderVar } from "../../Core/Program/ParameterBlock.js";
import { Aabb, buildLightBVH, kDefaultLightBVHOptions, type EmissiveTriangleInput, type LightBVHOptions } from "./LightBVHBuilder.js";
import { PackedNode, kInvalidCosConeAngle, type SharedNodeAttributes, type Vec3 } from "./LightBVHTypes.js";

/** Mirrors SolidAngleBoundMethod (LightBVHSamplerSharedDefinitions.slang). */
export const kSolidAngleBoundMethods: Record<string, number> = { BoxToCenter: 1, BoxToAverage: 2, Sphere: 3 };

/** Mirrors LightBVHSampler::Options (buildOptions nested like the native serialization). */
export interface LightBVHSamplerOptions {
    buildOptions: LightBVHOptions;
    useBoundingCone: boolean;
    useLightingCone: boolean;
    disableNodeFlux: boolean;
    useUniformTriangleSampling: boolean;
    solidAngleBoundMethod: number; // SolidAngleBoundMethod
}

export const kDefaultLightBVHSamplerOptions: LightBVHSamplerOptions = {
    buildOptions: kDefaultLightBVHOptions,
    useBoundingCone: true,
    useLightingCone: true,
    disableNodeFlux: false,
    useUniformTriangleSampling: true,
    solidAngleBoundMethod: kSolidAngleBoundMethods["Sphere"]!,
};

export class LightBVHSampler {
    private readonly nodes: Buffer;
    private readonly triangleIndices: Buffer;
    private readonly triangleBitmasks: Buffer;
    private readonly options: LightBVHSamplerOptions;
    /** CPU copy of the tree (native mNodes/mNodeIndices) so refit() can update it in place. */
    private readonly packedNodes: PackedNode[] = [];
    private readonly triangleIndexArray: Uint32Array;
    private readonly triangleCount: number;
    readonly valid: boolean;

    constructor(device: Device, triangles: EmissiveTriangleInput[], options: LightBVHSamplerOptions = kDefaultLightBVHSamplerOptions) {
        this.options = options;
        const result = buildLightBVH(triangles, options.buildOptions);
        this.valid = result.valid;
        this.triangleCount = triangles.length;
        this.triangleIndexArray = result.triangleIndices;
        const words = new Uint32Array(result.nodes);
        for (let i = 0; i < result.nodeCount; i++) {
            const node = new PackedNode();
            node.data.set(words.subarray(i * 8, i * 8 + 8));
            this.packedNodes.push(node);
        }

        const make = (name: string, data: ArrayBufferView | ArrayBuffer, structSize: number) => {
            const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
            const buf = new Buffer(device, {
                size: Math.max(bytes.byteLength, structSize),
                structSize,
                bindFlags: ResourceBindFlags.ShaderResource,
                memoryType: MemoryType.DeviceLocal,
                name: `LightBVH::${name}`,
            });
            buf.setBlob(bytes);
            return buf;
        };
        this.nodes = make("nodes", result.nodes, 32);
        this.triangleIndices = make("triangleIndices", result.triangleIndices, 4);
        this.triangleBitmasks = make("triangleBitmasks", result.triangleBitmasks, 8);
    }

    /**
     * Mirrors LightBVH::refit: keeps the tree structure and recomputes the node
     * bounds + normal cones bottom-up from the current emissive triangles (native
     * runs LightBVHRefit.cs.slang on the GPU; the web tree is CPU-resident so the
     * same per-node math runs here and the packed nodes are re-uploaded).
     * Returns false when the triangle set changed size (caller must rebuild).
     */
    refit(triangles: EmissiveTriangleInput[]): boolean {
        if (!this.valid || triangles.length !== this.triangleCount) return false;
        const nodes = this.packedNodes;
        const triIdx = this.triangleIndexArray;
        const f = Math.fround;
        const refitNode = (nodeIndex: number): SharedNodeAttributes => {
            const node = nodes[nodeIndex]!;
            const attribs = node.getNodeAttributes();
            if (node.isLeaf()) {
                // updateLeafNodes: AABB over the leaf's triangles, cone from the normal sum.
                const count = node.getLeafTriangleCount();
                const offset = node.getLeafTriangleOffset();
                const bounds = new Aabb();
                const normalsSum: Vec3 = [0, 0, 0];
                for (let i = 0; i < count; i++) {
                    const tri = triangles[triIdx[offset + i]!]!;
                    for (const p of tri.posW) bounds.includePoint(p);
                    for (let k = 0; k < 3; k++) normalsSum[k] = f(normalsSum[k]! + tri.normal[k]!);
                }
                const len = f(Math.hypot(normalsSum[0], normalsSum[1], normalsSum[2]));
                let cosConeAngle = kInvalidCosConeAngle;
                let coneDirection: Vec3 = [0, 0, 0];
                if (len >= 1.17549435e-38) {
                    coneDirection = [f(normalsSum[0] / len), f(normalsSum[1] / len), f(normalsSum[2] / len)];
                    cosConeAngle = 1;
                    for (let i = 0; i < count; i++) {
                        const n = triangles[triIdx[offset + i]!]!.normal;
                        const d = f(f(f(coneDirection[0] * n[0]!) + f(coneDirection[1] * n[1]!)) + f(coneDirection[2] * n[2]!));
                        cosConeAngle = Math.min(cosConeAngle, d);
                    }
                    cosConeAngle = Math.max(cosConeAngle, -1);
                }
                const updated: SharedNodeAttributes = { ...attribs, origin: bounds.center(), extent: halfExtent(bounds), cosConeAngle, coneDirection };
                node.setNodeAttributes(updated);
                return updated;
            }
            // updateInternalNodes: union of the (quantized) child attributes.
            const left = refitNode(nodeIndex + 1);
            const right = refitNode(node.getRightChildIdx());
            const bounds = new Aabb();
            for (const c of [left, right]) {
                bounds.includePoint([f(c.origin[0] - c.extent[0]), f(c.origin[1] - c.extent[1]), f(c.origin[2] - c.extent[2])]);
                bounds.includePoint([f(c.origin[0] + c.extent[0]), f(c.origin[1] + c.extent[1]), f(c.origin[2] + c.extent[2])]);
            }
            const union = coneUnionRefit(left, right);
            const updated: SharedNodeAttributes = { ...attribs, origin: bounds.center(), extent: halfExtent(bounds), cosConeAngle: union.cos, coneDirection: union.dir };
            node.setNodeAttributes(updated);
            return updated;
        };
        refitNode(0);
        const bytes = new Uint32Array(nodes.length * 8);
        nodes.forEach((n, i) => bytes.set(n.data, i * 8));
        this.nodes.setBlob(bytes);
        return true;
    }

    /** Node attributes of the CPU tree (tests / stats). */
    getNodeAttributes(nodeIndex: number): SharedNodeAttributes {
        return this.packedNodes[nodeIndex]!.getNodeAttributes();
    }

    getNodeCount(): number {
        return this.packedNodes.length;
    }

    /** Mirrors LightBVHSampler::getDefines. */
    getDefines(): DefineList {
        const o = this.options;
        return new DefineList()
            .add("_EMISSIVE_LIGHT_SAMPLER_TYPE", "1") // EMISSIVE_LIGHT_SAMPLER_LIGHT_BVH
            .add("_USE_BOUNDING_CONE", o.useBoundingCone ? "1" : "0")
            .add("_USE_LIGHTING_CONE", o.useLightingCone ? "1" : "0")
            .add("_DISABLE_NODE_FLUX", o.disableNodeFlux ? "1" : "0")
            .add("_USE_UNIFORM_TRIANGLE_SAMPLING", o.useUniformTriangleSampling ? "1" : "0")
            .add("_ACTUAL_MAX_TRIANGLES_PER_NODE", String(o.buildOptions.maxTriangleCountPerLeaf))
            .add("_SOLID_ANGLE_BOUND_METHOD", String(o.solidAngleBoundMethod));
    }

    /** Binds under emissiveSampler (fields live at _lightBVH.*). */
    bindShaderData(var_: ShaderVar): void {
        const bvh = var_["_lightBVH"] as ShaderVar;
        bvh["nodes"] = this.nodes;
        bvh["triangleIndices"] = this.triangleIndices;
        bvh["triangleBitmasks"] = this.triangleBitmasks;
    }
}

function halfExtent(b: Aabb): Vec3 {
    const f = Math.fround;
    return [f(f(b.max[0] - b.min[0]) * 0.5), f(f(b.max[1] - b.min[1]) * 0.5), f(f(b.max[2] - b.min[2]) * 0.5)];
}

/** Mirrors the cone union in LightBVHRefit.cs.slang::updateInternalNodes. */
function coneUnionRefit(left: SharedNodeAttributes, right: SharedNodeAttributes): { dir: Vec3; cos: number } {
    const f = Math.fround;
    const sum: Vec3 = [f(left.coneDirection[0] + right.coneDirection[0]), f(left.coneDirection[1] + right.coneDirection[1]), f(left.coneDirection[2] + right.coneDirection[2])];
    const len = f(Math.hypot(sum[0], sum[1], sum[2]));
    const dir: Vec3 = len > 0 ? [f(sum[0] / len), f(sum[1] / len), f(sum[2] / len)] : [0, 0, 0];
    let cos = kInvalidCosConeAngle;
    if (len >= 1.17549435e-38 && left.cosConeAngle !== kInvalidCosConeAngle && right.cosConeAngle !== kInvalidCosConeAngle) {
        const dot = (a: Vec3, b: Vec3) => f(f(f(a[0] * b[0]) + f(a[1] * b[1])) + f(a[2] * b[2]));
        const sinFromCos = (c: number) => f(Math.sqrt(Math.max(0, f(1 - f(c * c)))));
        const cosLeftDiff = dot(dir, left.coneDirection);
        const sinLeftDiff = sinFromCos(cosLeftDiff);
        const cosRightDiff = dot(dir, right.coneDirection);
        const sinRightDiff = sinFromCos(cosRightDiff);
        const sinLeftCone = sinFromCos(left.cosConeAngle);
        const sinRightCone = sinFromCos(right.cosConeAngle);
        const sinLeftTotal = f(f(sinLeftCone * cosLeftDiff) + f(sinLeftDiff * left.cosConeAngle));
        const sinRightTotal = f(f(sinRightCone * cosRightDiff) + f(sinRightDiff * right.cosConeAngle));
        if (sinLeftTotal > 0 && sinRightTotal > 0) {
            const cosLeftTotal = f(f(left.cosConeAngle * cosLeftDiff) - f(sinLeftCone * sinLeftDiff));
            const cosRightTotal = f(f(right.cosConeAngle * cosRightDiff) - f(sinRightCone * sinRightDiff));
            cos = Math.max(Math.min(cosLeftTotal, cosRightTotal), -1);
        }
    }
    return { dir, cos };
}
