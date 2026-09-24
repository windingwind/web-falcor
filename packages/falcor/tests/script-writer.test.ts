/** ScriptWriter (Utils/Scripting/ScriptWriter) and the getScript() output built on it. */
import { describe, expect, it } from "vitest";
import { ScriptWriter } from "../src/Utils/Scripting/ScriptWriter.js";
import { Clock } from "../src/Utils/Timing/Clock.js";
import { float3 } from "../src/Utils/Math/Vector.js";

describe("ScriptWriter", () => {
    it("writes Python calls and assignments with literal arguments", () => {
        expect(ScriptWriter.makeFunc("f")).toBe("f()\n");
        expect(ScriptWriter.makeMemberFunc("m", "loadScene", "a/b.pyscene", true, 2)).toBe('m.loadScene("a/b.pyscene", True, 2)\n');
        expect(ScriptWriter.makeSetProperty("m.scene.camera", "position", new float3(1, 2.5, -3))).toBe("m.scene.camera.position = float3(1, 2.5, -3)\n");
        expect(ScriptWriter.makeSetProperty("x", "d", { a: [1, 2], b: false })).toBe('x.d = {"a": [1, 2], "b": False}\n');
        expect(ScriptWriter.getPathString("C:\\\\data\\\\x.py")).toBe("C:/data/x.py");
    });

    it("writes the clock settings like Clock::getScript", () => {
        const clock = new Clock().setFramerate(60).setExitTime(4);
        clock.pause();
        expect(clock.getScript("m.clock")).toBe(
            "m.clock.time = 0\nm.clock.framerate = 60\nm.clock.exitTime = 4\n# If framerate is not zero, you can use the frame property to set the start frame\n# m.clock.frame = 0\nm.clock.pause()\n",
        );
        expect(clock.shouldExit()).toBe(false);
        clock.setTime(5);
        expect(clock.shouldExit()).toBe(true);
    });
});
