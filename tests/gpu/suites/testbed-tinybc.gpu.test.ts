/**
 * Falcor's TinyBC.py (Slang autodiff BC7 mode-6 encoder, scripts/python/TinyBC) run
 * unmodified: PIL loads the input, the encoder back-propagates through BC decoding, and
 * the script prints the decoded PSNR, which must match native's print at several step
 * counts. Native values: `python TinyBC.py tests/oracle/assets/tinybc-input.png [-s N]`
 * with Falcor's python module.
 */

import { initScripting, runTestbedScript } from "@web-falcor/falcor";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("Testbed.tinyBCScript", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const png = new Uint8Array(await (await fetch("/tests/oracle/assets/tinybc-input.png")).arrayBuffer());
    const native: [string[], string][] = [[["-s", "1"], "21.62"], [["-s", "20"], "33.41"], [[], "48.54"]];
    for (const [args, expected] of native) {
        const { stdout } = await runTestbedScript(device, "/Falcor/scripts/python/TinyBC/TinyBC.py", {
            extraFiles: ["BCTypes.slang"],
            files: { "input.png": png },
            argv: ["input.png", ...args],
        });
        const psnr = stdout.find((l) => l.startsWith("PSNR:"))?.slice(5).trim();
        console.error(`# tinybc ${args.join(" ") || "default"}: PSNR ${psnr} (native ${expected})`);
        expectEq(psnr, expected, `PSNR with ${args.join(" ") || "default steps"}`);
    }
});
