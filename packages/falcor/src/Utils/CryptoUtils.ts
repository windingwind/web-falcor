/** Mirrors Falcor/Utils/CryptoUtils (SHA1): incremental update(), finalize() and one-shot compute(). */
export class SHA1 {
    private h = new Uint32Array([0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476, 0xc3d2e1f0]);
    private block = new Uint8Array(64);
    private used = 0;
    private length = 0;
    private readonly w = new Uint32Array(80);

    /** Appends bytes (strings are UTF-8 encoded). */
    update(data: Uint8Array | string): this {
        const bytes = typeof data === "string" ? new TextEncoder().encode(data) : data;
        for (const b of bytes) {
            this.block[this.used++] = b;
            if (this.used === 64) this.processBlock();
        }
        this.length += bytes.length;
        return this;
    }

    /** The 20-byte message digest; resets the state like native. */
    finalize(): Uint8Array {
        const bits = this.length * 8;
        this.block[this.used++] = 0x80;
        if (this.used > 56) {
            this.block.fill(0, this.used);
            this.processBlock();
        }
        this.block.fill(0, this.used, 56);
        const view = new DataView(this.block.buffer);
        view.setUint32(56, Math.floor(bits / 0x100000000));
        view.setUint32(60, bits >>> 0);
        this.processBlock();
        const md = new Uint8Array(20);
        const out = new DataView(md.buffer);
        this.h.forEach((v, i) => out.setUint32(i * 4, v));
        this.h = new Uint32Array([0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476, 0xc3d2e1f0]);
        this.length = 0;
        return md;
    }

    static compute(data: Uint8Array | string): Uint8Array {
        return new SHA1().update(data).finalize();
    }

    /** Lowercase hex. Native SHA1::toString sets setw(2) once, so its later bytes lose leading zeros; this pads all. */
    static toString(md: Uint8Array): string {
        return Array.from(md, (b) => b.toString(16).padStart(2, "0")).join("");
    }

    private processBlock(): void {
        const w = this.w;
        const view = new DataView(this.block.buffer);
        for (let i = 0; i < 16; i++) w[i] = view.getUint32(i * 4);
        for (let i = 16; i < 80; i++) {
            const x = w[i - 3]! ^ w[i - 8]! ^ w[i - 14]! ^ w[i - 16]!;
            w[i] = (x << 1) | (x >>> 31);
        }
        let [a, b, c, d, e] = this.h as unknown as number[];
        for (let i = 0; i < 80; i++) {
            const [f, k] = i < 20 ? [(b! & c!) | (~b! & d!), 0x5a827999] : i < 40 ? [b! ^ c! ^ d!, 0x6ed9eba1] : i < 60 ? [(b! & c!) | (b! & d!) | (c! & d!), 0x8f1bbcdc] : [b! ^ c! ^ d!, 0xca62c1d6];
            const t = (((a! << 5) | (a! >>> 27)) + f + e! + k + w[i]!) >>> 0;
            [e, d, c, b, a] = [d, c, ((b! << 30) | (b! >>> 2)) >>> 0, a, t];
        }
        const h = this.h;
        h[0] = h[0]! + a!;
        h[1] = h[1]! + b!;
        h[2] = h[2]! + c!;
        h[3] = h[3]! + d!;
        h[4] = h[4]! + e!;
        this.used = 0;
    }
}
