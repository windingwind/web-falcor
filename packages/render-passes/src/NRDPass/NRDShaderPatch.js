/**
 * WGSL patches for the NRD 3.1.0 shaders, as plain JS so scripts/setup-web.mjs (Node) and NRDPass (browser)
 * share them. Bundled builds fetch NRD from its repo and patch at runtime: its licence forbids shipping the source.
 */

/** NRD's own repo at the tag Falcor pins, and its MathLib submodule's commit. */
export const kNRDUpstream = "https://raw.githubusercontent.com/NVIDIAGameWorks/RayTracingDenoiser/v3.1.0";
export const kMathLibUpstream = "https://raw.githubusercontent.com/NVIDIAGameWorks/MathLib/afb2c3f070bc15abac6ae1aad19847a848a67cbf";

/** Returns `text` patched for WGSL; `path` is relative to the NRD repo root ("Shaders/Include/NRD.hlsli"). */
export function patchNRDShader(path, text) {
    if (path === "Shaders/Include/NRD.hlsli") return patchRegisters(text);
    if (path.startsWith("Shaders/Resources/")) return patchResource(path.slice("Shaders/Resources/".length), text);
    if (path === "Shaders/Include/REBLUR/REBLUR_Common.hlsli") return patchReblurCommon(text);
    if (path.startsWith("Shaders/Include/REBLUR/") && path.includes("HistoryFix")) {
        return text.replace(/(gOut_\w+), (gIn_\w+), gIn_ScaledViewZ \)/g, "$1, $2, gIn_ScaledViewZ, $1__in )");
    }
    return text;
}

// FXC/DXC register bindings would collide in WGSL (t0/u0/s0 all map to binding 0).
function patchRegisters(src) {
    const start = src.indexOf("#elif( defined NRD_COMPILER_FXC || defined NRD_COMPILER_DXC )");
    const end = src.indexOf("#elif( defined NRD_COMPILER_PSSLC )");
    if (start < 0 || end < 0) throw new Error("NRD.hlsli: DXC macro block not found");
    const block = src.slice(start, end).replace("cbuffer globalConstants : register( b0 ) {", "cbuffer globalConstants {").replaceAll(": register( regName ## bindingIndex );", ";");
    return src.slice(0, start) + block + src.slice(end);
}

function patchResource(name, text) {
    // `RWTexture2D<unorm float4>` has no WGSL form; the bound unorm format clamps on store anyway.
    let patched = text.replaceAll("<unorm ", "<");
    // Read-modify-write outputs (gInOut_*): WGSL read_write storage takes only r32 formats, so each becomes a
    // write target plus a read copy NRDPass fills before the dispatch, behind a view with the original name.
    patched = patched.replace(/NRD_OUTPUT_TEXTURE\( RWTexture2D<([\w ]+)>, (gInOut_\w+), u, (\d+) \)/g, (_m, type, name, reg) =>
        [
            `NRD_OUTPUT_TEXTURE( RWTexture2D<${type}>, ${name}__out, u, ${reg} )`,
            `        Texture2D<${type}> ${name}__in; // WebFalcor: read copy of ${name}`,
            `        struct WebFalcorRW_${name} {`,
            `            __subscript(uint2 p) -> ${type} { get { return ${name}__in[p]; } [nonmutating] set { ${name}__out[p] = newValue; } }`,
            `            __subscript(int2 p) -> ${type} { get { return ${name}__in[p]; } [nonmutating] set { ${name}__out[p] = newValue; } }`,
            `        };`,
            `        static WebFalcorRW_${name} ${name};`,
        ].join("\n"),
    );
    // Output blocks with more than 8 storage textures (REBLUR's MipGen): WebGPU allows 8 per stage, so the mip
    // outputs (_x2/_x4/_x8) become storage buffers behind views with the original names; NRDPass copies them into mips.
    patched = patched.replace(/NRD_OUTPUT_TEXTURE_START([\s\S]*?)NRD_OUTPUT_TEXTURE_END/g, (block) => {
        if ((block.match(/NRD_OUTPUT_TEXTURE\(/g) ?? []).length <= 8) return block;
        return block.replace(/NRD_OUTPUT_TEXTURE\( RWTexture2D<(\w+)>, (\w+_x\d), u, (\d+) \)/g, (_m, type, name) => {
            const get = type === "float" ? ".x" : type === "float2" ? ".xy" : "";
            const set = type === "float" ? "float4(newValue, 0, 0, 0)" : type === "float2" ? "float4(newValue, 0, 0)" : "newValue";
            return [
                `RWStructuredBuffer<float4> ${name}__buf; // WebFalcor: buffer-backed ${name}`,
                `        struct WebFalcorBuf_${name} {`,
                `            __subscript(uint2 p) -> ${type} { get { return ${name}__buf[p.y * gWebFalcorBufStride + p.x]${get}; } [nonmutating] set { ${name}__buf[p.y * gWebFalcorBufStride + p.x] = ${set}; } }`,
                `            __subscript(int2 p) -> ${type} { get { return ${name}__buf[p.y * gWebFalcorBufStride + p.x]${get}; } [nonmutating] set { ${name}__buf[p.y * gWebFalcorBufStride + p.x] = ${set}; } }`,
                `        };`,
                `        static WebFalcorBuf_${name} ${name};`,
            ].join("\n");
        });
    });
    if (patched.includes("gWebFalcorBufStride")) patched = `uniform uint gWebFalcorBufStride; // WebFalcor: row stride of the buffer-backed outputs\n${patched}`;
    // REBLUR HistoryFix reads its float4 outputs' previous values in ReconstructHistory: same read copy, passed as an extra argument.
    if (name.includes("HistoryFix")) patched = patched.replace(/NRD_OUTPUT_TEXTURE\( RWTexture2D<float4>, (gOut_\w+), u, (\d+) \)/g, (m, out) => `${m}\n        Texture2D<float4> ${out}__in; // WebFalcor: read copy of ${out}`);
    return patched;
}

function patchReblurCommon(src) {
    const patched = src
        .replace("RWTexture2D<float4> texOut, Texture2D<float4> texIn, Texture2D<float> texScaledViewZ )", "RWTexture2D<float4> texOut, Texture2D<float4> texIn, Texture2D<float> texScaledViewZ, Texture2D<float4> texOutPrev )")
        .replace("    float4 c0 = texOut[ pixelPos ];", "    float4 c0 = texOutPrev[ pixelPos ];");
    if (patched === src) throw new Error("REBLUR_Common.hlsli: ReconstructHistory not found");
    return patched;
}
