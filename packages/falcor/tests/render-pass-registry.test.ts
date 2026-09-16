import { describe, expect, it } from "vitest";
import { getRegisteredRenderPasses, loadRenderPassLibrary } from "../src/RenderGraph/RenderPass.js";

describe("loadRenderPassLibrary", () => {
    it("imports a module that registers passes and reports the new types", async () => {
        const source = `globalThis.webFalcorPlugins.registerRenderPass("PluginTestPass", () => ({ name: "plugin" }));`;
        const added = await loadRenderPassLibrary(`data:text/javascript,${encodeURIComponent(source)}`);
        expect(added).toEqual(["PluginTestPass"]);
        expect(getRegisteredRenderPasses()).toContain("PluginTestPass");
        // Re-loading registers nothing new.
        expect(await loadRenderPassLibrary(`data:text/javascript,${encodeURIComponent(source)}`)).toEqual([]);
    });
});
