/**
 * InvalidPixelDetectionPass: NaN pixels red, Inf green, valid black
 * (unmodified upstream shader; direct pass execution over an injected
 * NaN/Inf texture).
 */

import { Properties, RenderData, ResourceFormat, Texture, ResourceType, ResourceBindFlags } from "@web-falcor/falcor";
import { InvalidPixelDetectionPass } from "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("InvalidPixelDetection.flagsNanAndInf", async ({ device }) => {
    const size = 4;
    const data = new Float32Array(size * size * 4).fill(1);
    data[0] = NaN; // pixel (0,0) -> red
    data[4] = Infinity; // pixel (1,0) -> green
    const src = device.createTexture2D(size, size, ResourceFormat.RGBA32Float, 1, 1, data);
    const dst = new Texture(device, {
        type: ResourceType.Texture2D,
        width: size,
        height: size,
        format: ResourceFormat.RGBA32Float,
        bindFlags: ResourceBindFlags.RenderTarget | ResourceBindFlags.ShaderResource,
        mipLevels: 1,
        name: "invalidPixel::dst",
    });

    const pass = new InvalidPixelDetectionPass(device, new Properties({}));
    pass.execute(device.renderContext, new RenderData(new Map([["src", src], ["dst", dst]]), [size, size]));

    const out = new Float32Array((await device.renderContext.readTextureSubresource(dst)).buffer);
    const px = (x: number, y: number) => [out[(y * size + x) * 4], out[(y * size + x) * 4 + 1], out[(y * size + x) * 4 + 2]];
    expectEq(JSON.stringify(px(0, 0)), JSON.stringify([1, 0, 0]), "NaN pixel red");
    expectEq(JSON.stringify(px(1, 0)), JSON.stringify([0, 1, 0]), "Inf pixel green");
    expectEq(JSON.stringify(px(2, 2)), JSON.stringify([0, 0, 0]), "valid pixel black");
});
