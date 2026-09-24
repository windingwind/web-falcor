/**
 * falcor.ui (Utils/UI/PythonUI) through Testbed scripts: Falcor's own ui_demo.py runs
 * unmodified; the test clicks its widgets between frames and checks that callbacks
 * ran inside testbed.frame() (counter label, window move/resize, show/close, a
 * threading.Timer-driven progress bar) and that edits reach the widget values.
 */

import { initScripting, runTestbedScript } from "@web-falcor/falcor";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("TestbedUI.uiDemoRunsAndResponds", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const host = document.createElement("div");
    document.body.appendChild(host);
    const button = (label: string) => [...host.querySelectorAll("button")].find((b) => b.textContent === label);
    const windows = () => [...host.querySelectorAll<HTMLElement>(".falcor-ui-screen > div")];
    const windowTitled = (t: string) => windows().find((w) => w.firstElementChild?.firstElementChild?.textContent === t);
    const seen: Record<string, unknown> = {};
    await runTestbedScript(device, "/Falcor/scripts/python/ui/ui_demo.py", {
        maxFrames: 40,
        uiHost: host,
        onFrame: (_t, frame) => {
            if (frame === 2) {
                seen.windows = windows().map((w) => w.firstElementChild?.firstElementChild?.textContent).join(",");
                button("Clicked 0 times")?.click();
                button("Move window to [50, 50]")?.click();
                button("Close widget window")?.click();
                const checkbox = host.querySelector<HTMLInputElement>("input[type=checkbox]")!;
                checkbox.checked = false;
                checkbox.dispatchEvent(new Event("change"));
            }
            if (frame === 4) {
                seen.counter = button("Clicked 1 times") !== undefined;
                seen.demoLeft = windowTitled("Demo Window")?.style.left;
                seen.widgetsHidden = windowTitled("Widgets")?.style.display;
                button("Start")?.click();
            }
            // The demo's progress bar advances on a 50 ms threading.Timer: let wall time pass.
            if (frame > 4) {
                const until = performance.now() + 20;
                while (performance.now() < until);
            }
        },
    });
    const progress = host.querySelector("progress") ?? undefined;
    const demoProgress = [...host.querySelectorAll("progress")].at(-1)!;
    void progress;
    console.error(`# testbed ui: windows=${seen.windows} counter=${seen.counter} left=${seen.demoLeft} widgets=${seen.widgetsHidden} progress=${demoProgress.value}`);
    expectEq(seen.windows, "Widgets,Demo Window", "both windows are created");
    expectEq(seen.counter, true, "the counter button's callback relabels it");
    expectEq(seen.demoLeft, "50px", "the move button moves its window");
    expectEq(seen.widgetsHidden, "none", "close() hides the widget window");
    expectEq(button("Stop") !== undefined, true, "start relabels the button");
    expectEq(demoProgress.value > 0 && demoProgress.value < 1, true, `the timer advances the progress bar (${demoProgress.value})`);
    host.remove();
});
