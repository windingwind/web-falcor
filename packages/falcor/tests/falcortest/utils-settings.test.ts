/**
 * Transplant of FalcorTest Utils/SettingsTests.cpp (Settings options, attribute filters, search paths).
 */

import { describe, expect, it } from "vitest";
import { Settings, SettingsTypeError, type SettingsValue } from "../../src/Utils/Settings.js";

// Native uses "/c" as the fake drive root off Windows; weakly_canonical is the identity here.
const C_DRIVE = "/c";

describe("SettingsTests", () => {
    it("Settings_OptionsIntBool", () => {
        const settings = new Settings();
        settings.addOptions({ TrueAsBool: true, TrueAsInt: 1, FalseAsBool: false, FalseAsInt: 0 });

        expect(settings.getOption("TrueAsBool", false)).toBe(true);
        expect(settings.getOption("TrueAsBool", 0)).toBe(1);
        expect(settings.getOption("TrueAsInt", false)).toBe(true);
        expect(settings.getOption("TrueAsInt", 0)).toBe(1);

        expect(settings.getOption("FalseAsBool", true)).toBe(false);
        expect(settings.getOption("FalseAsBool", 1)).toBe(0);
        expect(settings.getOption("FalseAsInt", true)).toBe(false);
        expect(settings.getOption("FalseAsInt", 1)).toBe(0);
    });

    it("Settings_OptionsNesting", () => {
        const settings = new Settings();
        settings.addOptions({ mogwai: { value: 17 } });
        expect(settings.getOption("mogwai:value", 0)).toBe(17);
    });

    it("Settings_OptionsTypes", () => {
        const settings = new Settings();
        settings.addOptions({ string: "string", float: 1, int: 2, bool: true, "int[2]": [1, 2] });

        expect(settings.getOption("string", "")).toBe("string");
        expect(settings.getOption("float", 0)).toBe(1);
        expect(settings.getOption("int", 0)).toBe(2);
        expect(settings.getOption("bool", false)).toBe(true);

        const result = settings.getOption("int[2]", [-1, -2]);
        expect(result[0]).toBe(1);
        expect(result[1]).toBe(2);

        expect(() => settings.getOption("string", 3)).toThrow(SettingsTypeError);
        expect(() => settings.getOption("int", "test")).toThrow(SettingsTypeError);
        expect(() => settings.getOption("int[2]", 0)).toThrow(SettingsTypeError);
    });

    it("Settings_OptionsOverride", () => {
        const settings = new Settings();

        settings.addOptions({ mogwai: { value: 17 } });
        expect(settings.getOption("mogwai:value", 0)).toBe(17);

        settings.addOptions({ mogwai: { string: "test" } });
        expect(settings.getOption("mogwai:value", 0)).toBe(17);
        expect(settings.getOption("mogwai:string", "foo")).toBe("test");

        settings.addOptions({ mogwai: { string: "test2" } });
        expect(settings.getOption("mogwai:value", 0)).toBe(17);
        expect(settings.getOption("mogwai:string", "foo")).toBe("test2");

        settings.addOptions({ mogwai: { string: 14 } });
        expect(settings.getOption("mogwai:value", 0)).toBe(17);
        expect(settings.getOption("mogwai:string", 0)).toBe(14);

        settings.addOptions({ mogwai: 2 });
        const options = settings.getOptions();
        expect("mogwai:value" in options).toBe(false);
        expect("mogwai:string" in options).toBe(false);
        expect(settings.getOption("mogwai", 0)).toBe(2);
    });

    it("Settings_AttributeAssign", () => {
        const settings = new Settings();
        settings.addFilteredAttributes({
            usdImporter: {
                verbosity: 5,
                motionEnable: false,
                "motionEnable.filter": ["/World/Tiger.*", true],
                deduplicateVerts: true,
                "deduplicateVerts.filter": "/World/Tiger_Fur.*",
                multiplyEmission: 3,
                "multiplyEmission.filter": ["lights/.*"],
            },
        });

        const shapes = ["/World/Tiger/Body", "/World/Tiger_Fur/back", "/World/Ground", "lights/dome"].map((name) => ({
            name,
            motionEnabled: true,
            deduplicateVerts: false,
            verbosity: 0,
            multiplyEmission: 1,
        }));
        for (const shape of shapes) {
            shape.motionEnabled = settings.getAttribute(shape.name, "usdImporter:motionEnable", shape.motionEnabled);
            shape.deduplicateVerts = settings.getAttribute(shape.name, "usdImporter:deduplicateVerts", shape.deduplicateVerts);
            shape.verbosity = settings.getAttribute(shape.name, "usdImporter:verbosity", shape.verbosity);
            shape.multiplyEmission = settings.getAttribute(shape.name, "usdImporter:multiplyEmission", shape.multiplyEmission);
        }

        expect(shapes[0]).toEqual({ name: "/World/Tiger/Body", motionEnabled: true, deduplicateVerts: false, verbosity: 5, multiplyEmission: 1 });
        expect(shapes[1]).toEqual({ name: "/World/Tiger_Fur/back", motionEnabled: true, deduplicateVerts: true, verbosity: 5, multiplyEmission: 1 });
        expect(shapes[2]).toEqual({ name: "/World/Ground", motionEnabled: false, deduplicateVerts: false, verbosity: 5, multiplyEmission: 1 });
        expect(shapes[3]).toEqual({ name: "lights/dome", motionEnabled: false, deduplicateVerts: false, verbosity: 5, multiplyEmission: 3 });
    });

    it("Settings_AttributeFilters", () => {
        const settings = new Settings();
        settings.addFilteredAttributes([
            { regex: ".*", attributes: { curves: { ShadingRate: 5 } } },
            { regex: "/World/Tiger_Fur.*", attributes: { curves: { ShadingRate: Math.fround(0.1) } } },
        ]);

        expect(settings.getAttribute("/World/Tiger_Mane/top", "curves:ShadingRate", 1)).toBe(5);
        expect(settings.getAttribute("/World/Tiger_Fur/back", "curves:ShadingRate", 1)).toBe(Math.fround(0.1));
    });

    // Both spellings of the search-path keys must behave identically.
    const updatePaths = (make: (kind: string, value: string) => Record<string, SettingsValue>) => {
        const settings = new Settings();

        settings.addOptions(make("standardsearchpath", `${C_DRIVE}/media`));
        expect(settings.getSearchDirectories("media")).toEqual([`${C_DRIVE}/media`]);

        settings.addOptions(make("standardsearchpath", `${C_DRIVE}/media/different`));
        expect(settings.getSearchDirectories("media")).toEqual([`${C_DRIVE}/media/different`]);

        settings.addOptions(make("standardsearchpath", `&;${C_DRIVE}/media/two`));
        expect(settings.getSearchDirectories("media")).toEqual([`${C_DRIVE}/media/different`, `${C_DRIVE}/media/two`]);

        settings.addOptions(make("searchpath", `&;${C_DRIVE}/media/three`));
        expect(settings.getSearchDirectories("media")).toEqual([`${C_DRIVE}/media/three`]);

        settings.addOptions(make("searchpath", `&;@;${C_DRIVE}/media/four`));
        expect(settings.getSearchDirectories("media")).toEqual([
            `${C_DRIVE}/media/three`,
            `${C_DRIVE}/media/different`,
            `${C_DRIVE}/media/two`,
            `${C_DRIVE}/media/four`,
        ]);
    };

    it("Settings_UpdatePathsColon", () => {
        updatePaths((kind, value) => ({ [`${kind}:media`]: value }));
    });

    it("Settings_UpdatePathsSeparate", () => {
        updatePaths((kind, value) => ({ [kind]: { media: value } }));
    });
});
