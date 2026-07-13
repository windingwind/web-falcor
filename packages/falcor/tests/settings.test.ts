import { describe, expect, it } from "vitest";
import { Settings } from "../src/Utils/Settings.js";

describe("Settings options", () => {
    it("flattens nested dicts with colons (native flattenDictionary)", () => {
        const s = new Settings();
        s.addOptions({ PipedOutput: { enable: true, cmd: "ffmpeg" }, flat: 3 });
        expect(s.getOption("PipedOutput:enable", false)).toBe(true);
        expect(s.getOption("PipedOutput:cmd", "")).toBe("ffmpeg");
        expect(s.getOption("flat", 0)).toBe(3);
        expect(s.getOption("missing", 7)).toBe(7);
    });

    it("later addOptions calls override earlier keys", () => {
        const s = new Settings();
        s.addOptions({ a: 1 });
        s.addOptions({ a: 2 });
        expect(s.getOption("a", 0)).toBe(2);
        s.clearOptions();
        expect(s.getOption("a", 0)).toBe(0);
    });
});

describe("Settings attribute filters", () => {
    it("applies regex-scoped attributes with full-match semantics", () => {
        const s = new Settings();
        s.addFilteredAttributes({ regex: "Fur.*", attributes: { foo: 4, bar: { foobar: 5 } } });
        expect(s.getAttribute("FurBall", "foo", 0)).toBe(4);
        expect(s.getAttribute("FurBall", "bar:foobar", 0)).toBe(5); // nested flatten
        expect(s.getAttribute("NoFur", "foo", 0)).toBe(0);
        expect(s.getAttribute("TheFurBall", "foo", 0)).toBe(0); // regex_match, not search
    });

    it("treats a dict without `attributes` as the attribute block", () => {
        const s = new Settings();
        s.addFilteredAttributes({ foo: 6 });
        expect(s.getAttribute("anything", "foo", 0)).toBe(6);
    });

    it("applies ordered filters with later overrides (array form)", () => {
        const s = new Settings();
        s.addFilteredAttributes([
            { regex: ".*", attributes: { a: 1 } },
            { regex: "special", attributes: { a: 2 } },
        ]);
        expect(s.getAttribute("plain", "a", 0)).toBe(1);
        expect(s.getAttribute("special", "a", 0)).toBe(2);
    });

    it("supports the deprecated `attr.filter` syntax", () => {
        const s = new Settings();
        s.addFilteredAttributes({ foo: 4, "foo.filter": "Fur.*" });
        expect(s.getAttribute("FurBall", "foo", 0)).toBe(4);
        expect(s.getAttribute("Ball", "foo", 0)).toBe(0);
    });

    it("supports the negated `attr.filter` form (apply everywhere except matches)", () => {
        const s = new Settings();
        s.addFilteredAttributes({ bar: 5, "bar.filter": ["Fur.*", true] });
        expect(s.getAttribute("Ball", "bar", 0)).toBe(5);
        expect(s.getAttribute("FurBall", "bar", 0)).toBe(0); // unapplied by null
    });

    it("clears filters", () => {
        const s = new Settings();
        s.addFilteredAttributes({ foo: 1 });
        s.clearFilteredAttributes();
        expect(s.getAttributes("x")).toEqual({});
    });
});
