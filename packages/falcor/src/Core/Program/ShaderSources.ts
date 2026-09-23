/**
 * Fetches the shader tree listed in shader-file-list.json (native: the deployed shader
 * directories) and installs a ProgramManager over it. Registry keys follow native's
 * layout: Falcor/* at the root, plus RenderPasses/*, Tests/* and Samples/*.
 */

import type { Device } from "../API/Device.js";
import { ProgramManager } from "./Program.js";
import { initSlang } from "./SlangCompiler.js";
import { Logger } from "../../Utils/Logger.js";

interface ShaderFileList {
    falcorFiles: string[];
    renderPassFiles: string[];
    localFiles: string[];
    testFiles?: string[];
    sampleFiles?: string[];
    externalFiles?: { path: string; url: string }[];
}

/** Path -> source for every listed shader (fetched with bounded concurrency). */
export async function fetchShaderSources(listUrl = "/packages/falcor/shaders/generated/shader-file-list.json"): Promise<Map<string, string>> {
    const list = (await (await fetch(listUrl)).json()) as ShaderFileList;
    const jobs: { url: string; key: string }[] = [
        ...list.falcorFiles.map((f) => ({ url: `/Falcor/Source/Falcor/${f}`, key: f })),
        ...list.renderPassFiles.map((f) => ({ url: `/Falcor/Source/${f}`, key: f })),
        ...list.localFiles.map((f) => ({ url: `/packages/falcor/shaders/${f}`, key: f })),
        ...(list.testFiles ?? []).map((f) => ({ url: `/Falcor/Source/Tools/FalcorTest/${f}`, key: f })),
        ...(list.sampleFiles ?? []).map((f) => ({ url: `/Falcor/Source/${f}`, key: f })),
        ...(list.externalFiles ?? []).map(({ path, url }) => ({ url, key: path })),
    ];
    const sources = new Map<string, string>();
    const missing: string[] = [];
    let next = 0;
    const worker = async () => {
        while (next < jobs.length) {
            const { url, key } = jobs[next++]!;
            const res = await fetch(url);
            if (res.ok) sources.set(key, await res.text());
            else missing.push(`${url} (${res.status})`);
        }
    };
    await Promise.all(Array.from({ length: 24 }, worker));
    if (missing.length > 0) Logger.warning(`shader registry: ${missing.length} files failed to fetch; first: ${missing.slice(0, 3).join(", ")}`);
    return sources;
}

/** Initializes slang-wasm and gives the device a ProgramManager over the shader tree. */
export async function initProgramSystem(device: Device, slangUrl = "/tools/slang-wasm/slang-wasm.js"): Promise<void> {
    const sources = await fetchShaderSources();
    await initSlang(slangUrl);
    device.setProgramManager(new ProgramManager(device, (p) => sources.get(p), [...sources.keys()]));
}
