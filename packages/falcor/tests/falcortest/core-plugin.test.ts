/**
 * Transplant of FalcorTest Core/PluginTests.cpp: two plugin base classes with different create signatures.
 */

import { describe, expect, it } from "vitest";
import { PluginManager, PluginObject, PluginRegistry } from "../../src/Core/Plugin.js";

abstract class PluginBaseA extends PluginObject {
    abstract getText(): string;
}
class PluginA1 extends PluginBaseA {
    static readonly kPluginType = "PluginA1";
    static readonly kPluginInfo = { desc: "This is PluginA1" };
    static create = (text: string) => new PluginA1(text);
    constructor(private readonly text: string) {
        super();
    }
    getText(): string {
        return this.text;
    }
}
class PluginA2 extends PluginBaseA {
    static readonly kPluginType = "PluginA2";
    static readonly kPluginInfo = { desc: "This is PluginA2" };
    static create = (text: string) => new PluginA2(text);
    private readonly text: string;
    constructor(text: string) {
        super();
        this.text = `${text}\n${text}`;
    }
    getText(): string {
        return this.text;
    }
}

// A create function without arguments.
abstract class PluginBaseB extends PluginObject {}
class PluginB1 extends PluginBaseB {
    static readonly kPluginType = "PluginB1";
    static readonly kPluginInfo = { name: "This is PluginB1", sequence: [1, 2, 4, 8] };
    static create = () => new PluginB1();
}
class PluginB2 extends PluginBaseB {
    static readonly kPluginType = "PluginB2";
    static readonly kPluginInfo = { name: "This is PluginB2", sequence: [2, 4, 8, 16] };
    static create = () => new PluginB2();
}

describe("PluginTests", () => {
    it("Plugin", () => {
        const pm = new PluginManager();

        expect(pm.hasClass(PluginBaseA, "PluginA1")).toBe(false);
        expect(pm.hasClass(PluginBaseA, "PluginA2")).toBe(false);
        {
            const registry = new PluginRegistry(pm, 0);
            registry.registerClass(PluginBaseA, PluginA1);
            registry.registerClass(PluginBaseA, PluginA2);
        }
        expect(pm.hasClass(PluginBaseA, "PluginA1")).toBe(true);
        expect(pm.hasClass(PluginBaseA, "PluginA2")).toBe(true);
        {
            const infos = pm.getInfos<{ desc: string }>(PluginBaseA);
            const has = (name: string, desc: string) => infos.some(([n, info]) => n === name && info.desc === desc);
            expect(has("PluginA1", "This is PluginA1") && has("PluginA2", "This is PluginA2") && infos.length === 2).toBe(true);
        }
        {
            const pluginA1 = pm.createClass<(text: string) => PluginBaseA>(PluginBaseA, "PluginA1", "Hello world");
            const pluginA2 = pm.createClass<(text: string) => PluginBaseA>(PluginBaseA, "PluginA2", "Hello world again");
            const pluginA3 = pm.createClass<(text: string) => PluginBaseA>(PluginBaseA, "PluginA3", "");
            expect(pluginA1).not.toBeNull();
            expect(pluginA2).not.toBeNull();
            expect(pluginA3).toBeNull();
            expect(pluginA1!.getPluginType()).toBe("PluginA1");
            expect((pluginA1!.getPluginInfo() as { desc: string }).desc).toBe("This is PluginA1");
            expect(pluginA1!.getText()).toBe("Hello world");
            expect(pluginA2!.getPluginType()).toBe("PluginA2");
            expect((pluginA2!.getPluginInfo() as { desc: string }).desc).toBe("This is PluginA2");
            expect(pluginA2!.getText()).toBe("Hello world again\nHello world again");
        }

        expect(pm.hasClass(PluginBaseB, "PluginB1")).toBe(false);
        expect(pm.hasClass(PluginBaseB, "PluginB2")).toBe(false);
        {
            const registry = new PluginRegistry(pm, 0);
            registry.registerClass(PluginBaseB, PluginB1);
            registry.registerClass(PluginBaseB, PluginB2);
        }
        expect(pm.hasClass(PluginBaseB, "PluginB1")).toBe(true);
        expect(pm.hasClass(PluginBaseB, "PluginB2")).toBe(true);
        {
            const infos = pm.getInfos<{ sequence: number[] }>(PluginBaseB);
            const has = (name: string, seq: number[]) => infos.some(([n, info]) => n === name && info.sequence.join() === seq.join());
            expect(has("PluginB1", [1, 2, 4, 8]) && has("PluginB2", [2, 4, 8, 16]) && infos.length === 2).toBe(true);
        }
        {
            const pluginB1 = pm.createClass<() => PluginBaseB>(PluginBaseB, "PluginB1");
            const pluginB2 = pm.createClass<() => PluginBaseB>(PluginBaseB, "PluginB2");
            const pluginB3 = pm.createClass<() => PluginBaseB>(PluginBaseB, "PluginB3");
            expect(pluginB1).not.toBeNull();
            expect(pluginB2).not.toBeNull();
            expect(pluginB3).toBeNull();
            expect(pluginB1!.getPluginType()).toBe("PluginB1");
            expect((pluginB1!.getPluginInfo() as { sequence: number[] }).sequence).toEqual([1, 2, 4, 8]);
            expect(pluginB2!.getPluginType()).toBe("PluginB2");
            expect((pluginB2!.getPluginInfo() as { sequence: number[] }).sequence).toEqual([2, 4, 8, 16]);
        }
    });
});
