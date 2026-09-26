/**
 * Transplant of FalcorTest Utils/PathResolvingTests.cpp (env-var expansion and search-path lists).
 */

import { describe, expect, it } from "vitest";
import { resolveEnvVariables, resolvePath, resolveSearchPaths, weaklyCanonical } from "../../src/Utils/PathResolving.js";

// Native's non-Windows fake drive roots; weakly_canonical is the identity for these.
const C_DRIVE = "/c";
const D_DRIVE = "/d";
const E_DRIVE = "/e";

describe("PathResolvingTests", () => {
    it("PathResolving_ResolveEnvVar", () => {
        const proxyResolver = (varName: string) => ({ VAR1: "Value1", VAR2: "Value2", VAR3: "Value3" })[varName];
        const resolve = (s: string) => resolveEnvVariables(s, proxyResolver).value;

        expect(resolve("test1")).toBe("test1");
        expect(resolve("${VAR1}")).toBe("Value1");
        expect(resolve("_${VAR1}")).toBe("_Value1");
        expect(resolve("_${VAR1}_")).toBe("_Value1_");
        expect(resolve("${VAR1}${VAR2}")).toBe("Value1Value2");
        expect(resolve("_${VAR1}${VAR2}")).toBe("_Value1Value2");
        expect(resolve("_${VAR1}_${VAR2}")).toBe("_Value1_Value2");
        expect(resolve("_${VAR1}_${VAR2}_")).toBe("_Value1_Value2_");
        expect(resolve("${VAR1}_${VAR2}_")).toBe("Value1_Value2_");
        expect(resolve("${VAR1}_${VAR2}")).toBe("Value1_Value2");
        expect(resolve("_${VAR1}${VAR2}_")).toBe("_Value1Value2_");
    });

    it("PathResolving_Basic", () => {
        const standard = [`${C_DRIVE}/standard/path`];
        const current = [`${C_DRIVE}/current/path`];
        let result;

        result = resolveSearchPaths(current, [`${C_DRIVE}/update/path/one`, `${D_DRIVE}/update/path/two`], standard);
        expect(result.invalid).toEqual([]);
        expect(result.resolved).toEqual([`${C_DRIVE}/update/path/one`, `${D_DRIVE}/update/path/two`]);

        result = resolveSearchPaths(current, [`${C_DRIVE}/update/path/one;${D_DRIVE}/update/path/two`], standard);
        expect(result.invalid).toEqual([]);
        expect(result.resolved).toEqual([`${C_DRIVE}/update/path/one`, `${D_DRIVE}/update/path/two`]);

        result = resolveSearchPaths(current, [`${C_DRIVE}/update/path/one;&;${D_DRIVE}/update/path/two;@;`], standard);
        expect(result.invalid).toEqual([]);
        expect(result.resolved).toEqual([`${C_DRIVE}/update/path/one`, `${C_DRIVE}/current/path`, `${D_DRIVE}/update/path/two`, `${C_DRIVE}/standard/path`]);

        result = resolveSearchPaths(current, [`${C_DRIVE}/update/path/one;&`, `${D_DRIVE}/update/path/two;@;`], standard);
        expect(result.invalid).toEqual([]);
        expect(result.resolved).toEqual([`${C_DRIVE}/update/path/one`, `${C_DRIVE}/current/path`, `${D_DRIVE}/update/path/two`, `${C_DRIVE}/standard/path`]);

        result = resolveSearchPaths(current, [`update/path/one;&;${D_DRIVE}/update/path/two;@;`], standard);
        expect(result.invalid).toEqual(["update/path/one"]);
        expect(result.resolved).toEqual([`${C_DRIVE}/current/path`, `${D_DRIVE}/update/path/two`, `${C_DRIVE}/standard/path`]);

        result = resolveSearchPaths(current, ["update/path/one;&;:/update/path/two;@;"], standard);
        expect(result.invalid).toEqual(["update/path/one", ":/update/path/two"]);
        expect(result.resolved).toEqual([`${C_DRIVE}/current/path`, `${C_DRIVE}/standard/path`]);
    });

    it("PathResolving_EnvVar", () => {
        const proxyResolver = (varName: string) => ({ FALCOR_MEDIA_LIBRARY: `${C_DRIVE}/Project/Media`, USERNAME: "jdoe" })[varName];
        const standard = [`${C_DRIVE}/standard/path`];
        const current = [`${C_DRIVE}/current/path`];

        const result = resolveSearchPaths(current, ["${FALCOR_MEDIA_LIBRARY}", `${C_DRIVE}/Users/\${USERNAME}/.falcor/media`], standard, proxyResolver);
        expect(result.invalid).toEqual([]);
        expect(result.resolved).toEqual([`${C_DRIVE}/Project/Media`, `${C_DRIVE}/Users/jdoe/.falcor/media`]);
    });
    it("PathResolving_resolvePath", () => {
        const known = [`${C_DRIVE}/Users/jdoe/settings.ini`, `${D_DRIVE}/Project/Media/cornellbox.obj`, `${D_DRIVE}/Project/Geometry/cornellbox.usd`, `${E_DRIVE}/Textures/checkers.exr`, `${D_DRIVE}/Project/Media/test.txt`, `${E_DRIVE}/Textures/test.txt`];
        const fileChecker = (path: string) => known.includes(weaklyCanonical(path));
        const searchPaths = [`${D_DRIVE}/Project/Media/`, `${E_DRIVE}/Textures/`];
        const cwd = `${D_DRIVE}/Project/Geometry/`;
        const resolve = (p: string) => resolvePath(searchPaths, cwd, p, fileChecker);

        expect(resolve("../Media/cornellbox.obj")).toBe(`${D_DRIVE}/Project/Media/cornellbox.obj`);
        expect(resolve("cornellbox.obj")).toBe(`${D_DRIVE}/Project/Media/cornellbox.obj`);
        expect(resolve("test.txt")).toBe(`${D_DRIVE}/Project/Media/test.txt`);
        expect(resolve("checkers.exr")).toBe(`${E_DRIVE}/Textures/checkers.exr`);
        expect(resolve(`${C_DRIVE}/Users/jdoe/settings.ini`)).toBe(`${C_DRIVE}/Users/jdoe/settings.ini`);
        expect(resolve("./checkers.exr")).toBe("");
        expect(resolve("./test.txt")).toBe("");
        expect(resolve("./cornellbox.usd")).toBe(`${D_DRIVE}/Project/Geometry/cornellbox.usd`);
        expect(resolve("cornellbox.usd")).toBe("");
    });
});
