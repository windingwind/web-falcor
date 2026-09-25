/** Transplant of FalcorTest Utils/UnionFindTests.cpp: UnionFind against a trivial set-list reference. */

import { describe, expect, it } from "vitest";
import { UnionFind } from "../../src/Utils/Algorithm/UnionFind.js";
import { Mt19937 } from "../../src/Utils/SampleGenerators/CPUSampleGenerator.js";

class TrivialUnionFind {
    private sets: Set<number>[] = [];
    reset(size: number): void {
        this.sets = Array.from({ length: size }, (_, i) => new Set([i]));
    }
    findSet(v: number): number {
        const i = this.sets.findIndex((s) => s.has(v));
        return i < 0 ? 0 : i;
    }
    connectedSets(v0: number, v1: number): boolean {
        return this.findSet(v0) === this.findSet(v1);
    }
    unionSet(v0: number, v1: number): void {
        v0 = this.findSet(v0);
        v1 = this.findSet(v1);
        if (v0 === v1) return;
        for (const x of this.sets[v1]!) this.sets[v0]!.add(x);
        this.sets.splice(v1, 1);
    }
    getSetCount(): number {
        return this.sets.length;
    }
}

describe("UnionFindTests", () => {
    it("UnionFind_randomized", () => {
        const count = 10;
        for (let run = 0; run < 20; run++) {
            const r = new Mt19937(1234 + run);
            const uf = new UnionFind();
            const reference = new TrivialUnionFind();
            uf.reset(count);
            reference.reset(count);
            for (let iter = 0; reference.getSetCount() > 1 && iter < 1000; iter++) {
                const v0 = r.next() % count;
                const v1 = r.next() % count;
                expect(uf.connectedSets(v0, v1), `iter ${iter}/${run}`).toBe(reference.connectedSets(v0, v1));
                uf.unionSet(v0, v1);
                reference.unionSet(v0, v1);
                expect(uf.getSetCount(), `iter ${iter}/${run}`).toBe(reference.getSetCount());
                for (let i = 0; i < count; i++)
                    for (let j = i + 1; j < count; j++) expect(uf.connectedSets(i, j)).toBe(reference.connectedSets(i, j));
            }
        }
    });
});
