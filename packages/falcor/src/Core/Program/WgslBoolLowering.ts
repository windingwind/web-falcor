/**
 * WGSL post-passes for Slang's WGSL backend. Bools: `bool` is not host-shareable, yet
 * Slang emits bool members in the std140/std430 layout structs it generates for
 * cbuffers and structured buffers (WGSL rejects the module). HLSL stores a bool
 * as a 32-bit value, so those members become u32 and each access converts:
 * reads compare against zero, writes select 0/1. Lets upstream shaders with
 * bools in constant buffers compile unmodified.
 */

interface BoolMember {
    /** 1 for scalar bool, N for vecN<bool>. */
    width: number;
}

const kStructRe = /struct\s+(\w+)\s*\{([^}]*)\}/g;
const kMemberRe = /(@align\(\d+\)\s*)?(\w+)\s*:\s*(bool|vec([234])<bool>)\s*,/g;

/** Finds the start of the access chain (identifiers, `.`, balanced `[]`) ending just before `end`. */
function chainStart(src: string, end: number): number {
    let i = end;
    for (;;) {
        // `[...]` / `(...)` groups directly before the current position (index, deref, call)
        while (i > 0 && (src[i - 1] === "]" || src[i - 1] === ")")) {
            const close = src[i - 1]!;
            const open = close === "]" ? "[" : "(";
            let depth = 0;
            let k = i - 1;
            for (; k >= 0; k--) {
                if (src[k] === close) depth++;
                else if (src[k] === open && --depth === 0) break;
            }
            i = k;
        }
        let j = i;
        while (j > 0 && /\w/.test(src[j - 1]!)) j--;
        if (j === i) return i;
        i = j;
        if (i > 0 && src[i - 1] === ".") {
            i--;
            continue;
        }
        return i;
    }
}

/** Index just past the statement expression starting at `from` (up to `;` at depth 0). */
function expressionEnd(src: string, from: number): number {
    let depth = 0;
    for (let i = from; i < src.length; i++) {
        const c = src[i]!;
        if (c === "(" || c === "[") depth++;
        else if (c === ")" || c === "]") depth--;
        else if (c === ";" && depth === 0) return i;
    }
    return src.length;
}

export function lowerHostShareableBools(wgsl: string): string {
    const members = new Map<string, BoolMember>();
    const otherMembers = new Set<string>();
    for (const m of wgsl.matchAll(kStructRe)) {
        const hostShared = /_std(140|430)(_\d+)?$/.test(m[1]!);
        for (const f of m[2]!.matchAll(/(\w+)\s*:\s*[^,]+,/g)) if (!hostShared) otherMembers.add(f[1]!);
        if (!hostShared) continue;
        for (const f of m[2]!.matchAll(kMemberRe)) members.set(f[2]!, { width: f[4] ? Number(f[4]) : 1 });
    }
    // Only rename members whose names are unique to the layout structs.
    for (const name of [...members.keys()]) if (otherMembers.has(name)) members.delete(name);
    if (members.size === 0) return wgsl;

    // 1. Declarations: bool -> u32 inside the layout structs.
    let out = wgsl.replace(kStructRe, (whole, name: string, body: string) => {
        if (!/_std(140|430)(_\d+)?$/.test(name)) return whole;
        const newBody = body.replace(kMemberRe, (decl: string, align: string | undefined, member: string, _t: string, n: string | undefined) =>
            members.has(member) ? `${align ?? ""}${member} : ${n ? `vec${n}<u32>` : "u32"},` : decl,
        );
        return whole.replace(body, newBody);
    });

    // 2. Accesses, rewritten back to front so indices stay valid.
    const names = [...members.keys()].map((n) => n.replace(/[$]/g, "\\$&")).join("|");
    const accessRe = new RegExp(`\\.(${names})\\b(\\.[xyzw]{1,4}\\b)?`, "g");
    const matches = [...out.matchAll(accessRe)];
    for (let k = matches.length - 1; k >= 0; k--) {
        const m = matches[k]!;
        const start = chainStart(out, m.index!);
        if (start === m.index) continue; // not a member access
        const end = m.index! + m[0].length;
        const chain = out.slice(start, end);
        const swizzle = m[2] ? m[2].length - 1 : 0;
        const width = swizzle || members.get(m[1]!)!.width;
        const zero = width === 1 ? "0u" : `vec${width}<u32>(0u)`;
        const after = out.slice(end);
        const assign = /^\s*=(?!=)/.exec(after);
        if (assign) {
            const rhsStart = end + assign[0].length;
            const rhsEnd = expressionEnd(out, rhsStart);
            const rhs = out.slice(rhsStart, rhsEnd).trim();
            const one = width === 1 ? "1u" : `vec${width}<u32>(1u)`;
            out = `${out.slice(0, rhsStart)} select(${zero}, ${one}, ${rhs})${out.slice(rhsEnd)}`;
        } else {
            out = `${out.slice(0, start)}(${chain} != ${zero})${out.slice(end)}`;
        }
    }
    return out;
}

/**
 * HLSL wave intrinsics operate on the active lanes, including inside divergent
 * branches; WGSL's uniformity analysis rejects subgroup builtins there unless
 * the subgroup_uniformity diagnostic is turned off, which restores HLSL semantics.
 */
export function relaxSubgroupUniformity(wgsl: string): string {
    if (!/\bsubgroup[A-Z]\w*\s*\(/.test(wgsl) || /diagnostic\s*\(\s*off\s*,\s*subgroup_uniformity/.test(wgsl)) return wgsl;
    // Directives must precede declarations: insert after any leading `enable ...;` lines.
    const m = /^(\s*(?:enable[^;]*;\s*)*)/.exec(wgsl)!;
    return `${m[1]}diagnostic(off, subgroup_uniformity);\n${wgsl.slice(m[1]!.length)}`;
}

/** Storage formats WGSL allows read_write access to (core, without texture-format tiers). */
const kReadWriteFormats = new Set(["r32float", "r32uint", "r32sint"]);

/**
 * WGSL allows read_write storage textures only in r32 formats, but Slang emits every
 * RWTexture as read_write. A texture the module never reads is write-only in effect, so
 * its access becomes `write` (what the per-shader overrides do by hand). Textures that
 * are read keep read_write, and WebGPU reports them.
 */
export function lowerWriteOnlyStorageTextures(wgsl: string): string {
    return wgsl.replace(/(var\s+(\w+)\s*:\s*texture_storage_\w+<\s*(\w+)\s*,\s*)read_write(\s*>)/g, (m, pre: string, name: string, format: string, post: string) => {
        if (kReadWriteFormats.has(format)) return m;
        const read = new RegExp(`textureLoad\\s*\\(\\s*${name}\\b`).test(wgsl);
        return read ? m : `${pre}write${post}`;
    });
}
