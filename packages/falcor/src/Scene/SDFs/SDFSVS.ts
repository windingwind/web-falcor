/**
 * Sparse-voxel-set SDF grid host mirroring Scene/SDFs/SparseVoxelSet/
 * SDFSVS.cpp + SDFSVSVoxelizer.cs.slang. The native voxelizer is a compute
 * kernel emitting one AABB + one packed voxel per surface voxel; this ports it
 * to the CPU (the runtime SDFSVS.slang is consumed UNMODIFIED, so the host
 * emits exactly the resources the kernel would have). One AABB per surface
 * voxel means tens of thousands of primitives — practical only through the
 * SDF primitive-AABB BVH (Scene/SoftwareRT/Bvh.ts buildAabbBvh).
 */

import { generateCheeseCornerValues } from "./NDSDFGrid.js";

const kSqrt3 = Math.sqrt(3);

/** One voxel's world-space AABB. */
export interface VoxelAABB {
    min: [number, number, number];
    max: [number, number, number];
}

export class SDFSVS {
    gridWidth = 0;
    /** snorm8 dense field, (gridWidth+1)^3, x-fastest. */
    private sdField = new Int8Array(0);

    aabbs: VoxelAABB[] = [];
    /** Packed voxel records, 80 bytes each (4x uint4 slices + validity + pad). */
    voxelData = new Uint32Array(0);
    voxelCount = 0;

    get normalizationFactor(): number {
        return (0.5 * kSqrt3) / this.gridWidth;
    }

    /** Mirrors SDFGrid::generateCheeseValues + SDFSVS::setValuesInternal. */
    generateCheeseValues(gridWidth: number, seed: number): void {
        this.setValues(generateCheeseCornerValues(gridWidth, seed), gridWidth);
        this.voxelize();
    }

    /** Mirrors SDFSVS::setValuesInternal (snorm8, normFactor 2*gw/sqrt3). */
    setValues(cornerValues: Float32Array, gridWidth: number): void {
        this.gridWidth = gridWidth;
        const gwv = gridWidth + 1;
        this.sdField = new Int8Array(gwv * gwv * gwv);
        const normalizationMultiplier = (2 * gridWidth) / kSqrt3;
        for (let v = 0; v < this.sdField.length; v++) {
            const normalized = Math.min(Math.max(cornerValues[v]! * normalizationMultiplier, -1), 1);
            const integerScale = normalized * 127;
            this.sdField[v] = Math.trunc(integerScale >= 0 ? integerScale + 0.5 : integerScale - 0.5);
        }
    }

    /** snorm8 byte at a grid coord; out-of-[0,gridWidth) reads the 1.0 sentinel
     *  (mirrors the voxelizer's safeLoadValue: coord >= gridWidth -> 1.0). */
    private byteAt(x: number, y: number, z: number): number {
        const gw = this.gridWidth;
        if (x < 0 || y < 0 || z < 0 || x >= gw || y >= gw || z >= gw) return 127; // snorm8(1.0)
        const gwv = gw + 1;
        return this.sdField[x + gwv * (y + gwv * z)]! & 0xff;
    }

    /** True if the voxel at (vx,vy,vz) straddles the surface (its 8 corners). */
    private voxelContainsSurface(vx: number, vy: number, vz: number): boolean {
        const gw = this.gridWidth;
        if (vx < 0 || vy < 0 || vz < 0 || vx >= gw || vy >= gw || vz >= gw) return false;
        let anyNeg = false;
        let anyPos = false;
        for (let c = 0; c < 8; c++) {
            // int8 decode: sdField is stored signed; the corner value's sign.
            const b = this.byteAt(vx + (c & 1), vy + ((c >> 1) & 1), vz + ((c >> 2) & 1));
            const v = b < 128 ? b : b - 256; // int8
            if (v <= 0) anyNeg = true;
            if (v >= 0) anyPos = true;
        }
        return anyNeg && anyPos;
    }

    /** Mirrors SDFSVSVoxelizer: emit one AABB + packed voxel per surface voxel. */
    private voxelize(): void {
        const gw = this.gridWidth;
        const aabbs: VoxelAABB[] = [];
        const voxels: number[] = []; // 20 uint32 per voxel (80 bytes)

        for (let vz = 0; vz < gw; vz++) {
            for (let vy = 0; vy < gw; vy++) {
                for (let vx = 0; vx < gw; vx++) {
                    if (!this.voxelContainsSurface(vx, vy, vz)) continue;

                    // AABB: [-0.5 + vc/gw, -0.5 + (vc+1)/gw] (voxelizer form).
                    const pMin: [number, number, number] = [(vx - gw * 0.5) / gw, (vy - gw * 0.5) / gw, (vz - gw * 0.5) / gw];
                    const pMax: [number, number, number] = [(vx + 1 - gw * 0.5) / gw, (vy + 1 - gw * 0.5) / gw, (vz + 1 - gw * 0.5) / gw];
                    aabbs.push({ min: pMin, max: pMax });

                    // packedValuesSlices[s][yc] bits[8*zc] = byte at (vx+s-1, vy+yc-1, vz+zc-1).
                    const rec = new Array<number>(20).fill(0);
                    for (let s = 0; s < 4; s++) {
                        for (let yc = 0; yc < 4; yc++) {
                            let packed = 0;
                            for (let zc = 0; zc < 4; zc++) {
                                packed |= this.byteAt(vx + s - 1, vy + yc - 1, vz + zc - 1) << (8 * zc);
                            }
                            rec[s * 4 + yc] = packed >>> 0;
                        }
                    }
                    rec[16] = this.neighborValidityMask(vx, vy, vz);
                    // rec[17..19] = uint4 padding (validNeighborsMask is @align(16)).
                    for (const u of rec) voxels.push(u);
                }
            }
        }

        this.voxelCount = aabbs.length;
        this.aabbs = aabbs;
        this.voxelData = new Uint32Array(voxels);
    }

    /** Mirrors createNeighborValidityMask (3x3x3 neighbor surface containment). */
    private neighborValidityMask(vx: number, vy: number, vz: number): number {
        let mask = 0;
        for (let z = 0; z <= 2; z++) {
            for (let y = 0; y <= 2; y++) {
                for (let x = 0; x <= 2; x++) {
                    if (this.voxelContainsSurface(vx + x - 1, vy + y - 1, vz + z - 1)) {
                        mask |= 1 << (z + 3 * (y + 3 * x));
                    }
                }
            }
        }
        return mask >>> 0;
    }
}
