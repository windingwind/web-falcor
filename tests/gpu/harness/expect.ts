/** gtest-style EXPECT_* collection for transplanted FalcorTest tests: reports the first few failures. */

import { expectEq } from "./registry.js";
import { Mt19937, canonicalFloat } from "@web-falcor/falcor";

export class Expect {
    failures: string[] = [];
    count = 0;
    check(ok: boolean, msg: () => string): void {
        this.count++;
        if (!ok) this.failures.push(this.failures.length < 5 ? msg() : "");
    }
    done(what: string): void {
        expectEq(this.failures.length, 0, `${what}: ${this.failures.length} of ${this.count} checks failed; first: ${this.failures.slice(0, 5).join("; ")}`);
    }
}

/** libstdc++ std::uniform_real_distribution<float>(a, b) over an mt19937. */
export function uniformFloat(rng: Mt19937, a = 0, b = 1): () => number {
    return () => Math.fround(Math.fround(canonicalFloat(rng) * Math.fround(b - a)) + a);
}
