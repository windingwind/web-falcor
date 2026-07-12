import { describe, expect, it } from "vitest";
import { encodeExr } from "../src/Utils/Image/EXREncoder.js";
import { decodeExr } from "../src/Utils/Image/EXRDecoder.js";

describe("EXREncoder", () => {
    it("round-trips RGBA float32 bit-exactly through decodeExr", () => {
        const w = 5;
        const h = 3;
        const src = new Float32Array(w * h * 4);
        for (let i = 0; i < src.length; i++) src[i] = Math.fround(Math.sin(i * 1.7) * 10 + i * 0.001);
        const bytes = encodeExr(src, w, h);
        const { data, width, height } = decodeExr(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer);
        expect(width).toBe(w);
        expect(height).toBe(h);
        expect(Array.from(data)).toEqual(Array.from(src));
    });
});
