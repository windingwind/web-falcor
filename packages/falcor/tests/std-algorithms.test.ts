/**
 * StdAlgorithms.stdSort against libstdc++'s std::sort (fixtures/std-sort-reference.cpp): the same
 * index permutations for inputs full of ties, where an unstable sort's choices show.
 */

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { nthElement, stdSort } from "../src/Utils/Algorithm/StdAlgorithms.js";

describe("stdSort", () => {
    it("orders ties exactly as libstdc++ std::sort and std::nth_element", () => {
        const reference = readFileSync(new URL("./fixtures/std-sort-reference.txt", import.meta.url), "utf8").trim().split("\n");
        let s = 12345;
        const next = () => {
            s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
            return s >>> 8;
        };
        for (const line of reference.filter((l) => !l.startsWith("nth"))) {
            const [n, list] = line.split(":");
            const w = Float32Array.from({ length: Number(n) }, () => next() % 7);
            const p = Array.from({ length: w.length }, (_, i) => i);
            stdSort(p, (a, b) => w[a]! < w[b]!);
            expect(`${n}:${p.join(",")},`).toBe(line);
            expect(list!.length).toBeGreaterThan(0);
        }
        // nth_element lines follow, drawing from the same generator.
        let current: Float32Array | null = null;
        for (const line of reference.filter((l) => l.startsWith("nth"))) {
            const [, n, k] = line.split(":")[0]!.split(" ").map(Number) as [number, number, number];
            if (k === 0) current = Float32Array.from({ length: n }, () => next() % 5);
            const w = current!;
            const p = Array.from({ length: n }, (_, i) => i);
            nthElement(p, 0, k, n, (a, b) => w[a]! < w[b]!);
            expect(`nth ${n} ${k}:${p.join(",")},`).toBe(line);
        }
    });
});
