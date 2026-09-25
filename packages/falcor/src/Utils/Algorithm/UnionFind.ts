/** Mirrors Falcor/Utils/Algorithm/UnionFind.h: disjoint sets with path compression and union by size. */
export class UnionFind {
    private parent: number[] = [];
    private setSize: number[] = [];
    private setCount = 0;

    constructor(size = 0) {
        this.reset(size);
    }

    reset(size: number): void {
        this.parent = Array.from({ length: size }, (_, i) => i);
        this.setSize = new Array<number>(size).fill(1);
        this.setCount = size;
    }

    findSet(v: number): number {
        // Relink to the root so the search chain shortens.
        if (v === this.parent[v]) return v;
        return (this.parent[v] = this.findSet(this.parent[v]!));
    }

    connectedSets(v0: number, v1: number): boolean {
        return this.findSet(v0) === this.findSet(v1);
    }

    unionSet(v0: number, v1: number): void {
        v0 = this.findSet(v0);
        v1 = this.findSet(v1);
        if (v0 === v1) return;
        // The smaller set is parented under the larger one.
        if (this.setSize[v0]! < this.setSize[v1]!) [v0, v1] = [v1, v0];
        this.parent[v1] = v0;
        this.setSize[v0]! += this.setSize[v1]!;
        this.setCount--;
    }

    getSetCount(): number {
        return this.setCount;
    }
}
