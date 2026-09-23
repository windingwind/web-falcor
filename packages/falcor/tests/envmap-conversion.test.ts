/**
 * pbrt-v4 environment maps: the PFM decoder and the equal-area octahedral ->
 * lat-long conversion (EnvMapConverter).
 */

import { describe, expect, it } from "vitest";
import { decodePfm, isPfm } from "../src/Utils/Image/PFMDecoder.js";
import { convertEqualAreaOctToLatLong, latlongMapToWorld, ndirToOctEqualAreaUnorm } from "../src/Scene/Lights/EnvMapConverter.js";

function encodePfm(width: number, height: number, rows: number[][], channels: 1 | 3, littleEndian: boolean): Uint8Array {
    const header = new TextEncoder().encode(`${channels === 3 ? "PF" : "Pf"}\n${width} ${height}\n${littleEndian ? "-1.0" : "1.0"}\n`);
    const body = new DataView(new ArrayBuffer(width * height * channels * 4));
    let o = 0;
    for (const row of rows) for (const v of row) (body.setFloat32(o, v, littleEndian), (o += 4));
    const out = new Uint8Array(header.length + body.byteLength);
    out.set(header, 0);
    out.set(new Uint8Array(body.buffer), header.length);
    return out;
}

describe("PFM", () => {
    it("decodes bottom-to-top rows in either byte order", () => {
        for (const littleEndian of [true, false]) {
            // Stored rows: first the bottom row (1, 2), then the top row (3, 4).
            const bytes = encodePfm(2, 2, [[1, 1, 1, 2, 2, 2], [3, 3, 3, 4, 4, 4]], 3, littleEndian);
            expect(isPfm(bytes)).toBe(true);
            const image = decodePfm(bytes);
            expect(image.width).toBe(2);
            expect([image.data[0], image.data[4], image.data[8], image.data[12]]).toEqual([3, 4, 1, 2]);
            expect(image.data[3]).toBe(1);
        }
    });

    it("expands grayscale to rgb", () => {
        const image = decodePfm(encodePfm(1, 1, [[0.25]], 1, true));
        expect([...image.data]).toEqual([0.25, 0.25, 0.25, 1]);
    });

    it("rejects other formats", () => {
        expect(isPfm(new TextEncoder().encode("#?RADIANCE\n"))).toBe(false);
        expect(() => decodePfm(new TextEncoder().encode("P6\n1 1\n255\n"))).toThrow(/signature/);
    });
});

describe("equal-area octahedral mapping", () => {
    it("puts +z at the centre and -z at the corners", () => {
        expect(ndirToOctEqualAreaUnorm([0, 0, 1])).toEqual([0.5, 0.5]);
        expect(ndirToOctEqualAreaUnorm([0, 0, -1])).toEqual([1, 1]);
        const px = ndirToOctEqualAreaUnorm([1, 0, 0]);
        expect(px[0]).toBeCloseTo(1, 12);
        expect(px[1]).toBeCloseTo(0.5, 12);
        const py = ndirToOctEqualAreaUnorm([0, 1, 0]);
        expect(py[0]).toBeCloseTo(0.5, 12);
        expect(py[1]).toBeCloseTo(1, 12);
    });

    it("maps the lat-long centre to -z", () => {
        const d = latlongMapToWorld(0.5, 0.5);
        expect(d[0]).toBeCloseTo(0, 12);
        expect(d[1]).toBeCloseTo(0, 12);
        expect(d[2]).toBeCloseTo(-1, 12);
    });

    it("converts hemispheres into the right half of the lat-long map", () => {
        // The inner diamond of an octahedral map is the +z hemisphere.
        const n = 32;
        const data = new Float32Array(n * n * 4);
        for (let y = 0; y < n; y++) {
            for (let x = 0; x < n; x++) {
                const u = (x + 0.5) / n;
                const v = (y + 0.5) / n;
                const upper = Math.abs(2 * u - 1) + Math.abs(2 * v - 1) <= 1;
                data.set([upper ? 1 : 0, upper ? 0 : 1, 0.5, 1], (y * n + x) * 4);
            }
        }
        const latlong = convertEqualAreaOctToLatLong({ width: n, height: n, data });
        expect(latlong.width).toBe(2 * n);
        expect(latlong.height).toBe(n);
        let checked = 0;
        for (let y = 0; y < latlong.height; y++) {
            for (let x = 0; x < latlong.width; x++) {
                const dir = latlongMapToWorld((x + 0.5) / latlong.width, (y + 0.5) / latlong.height);
                // Skip the horizon band: |z| < 0.35 is within ~3 texels of the
                // diamond edge, where bilinear filtering blends the two halves.
                if (Math.abs(dir[2]) < 0.35) continue;
                const i = (y * latlong.width + x) * 4;
                expect(latlong.data[i]).toBeCloseTo(dir[2] > 0 ? 1 : 0, 5);
                expect(latlong.data[i + 2]).toBeCloseTo(0.5, 6);
                expect(latlong.data[i + 3]).toBe(0);
                checked++;
            }
        }
        expect(checked).toBeGreaterThan(latlong.width * latlong.height * 0.4);
    });

    it("requires a square source", () => {
        expect(() => convertEqualAreaOctToLatLong({ width: 4, height: 2, data: new Float32Array(32) })).toThrow(/square/);
    });
});
