/**
 * Local media (LocalMedia.ts): picked files served under a virtual path through the fetch hook, the
 * synchronous readers Python uses, and the directory listings the asset resolver and glob use.
 */

import { beforeAll, describe, expect, it, vi } from "vitest";
import { getLocalFile, isLocalMediaUrl, listLocalDirectory, mountLocalFiles, readLocalFileSync, unmountLocalFiles, type LocalMount } from "../src/Core/LocalMedia.js";
import { fetchDirectoryListing } from "../src/Core/AssetResolver.js";

const served = vi.fn(async () => new Response("served"));
let mount: LocalMount;

beforeAll(async () => {
    globalThis.fetch = served as unknown as typeof fetch;
    mount = await mountLocalFiles([
        ["Arcade/Arcade.pyscene", new File(["sceneBuilder.importScene('Arcade.fbx')"], "Arcade.pyscene")],
        ["Arcade/Arcade.fbx", new File([new Uint8Array([1, 2, 3])], "Arcade.fbx")],
        ["Arcade/Textures/wood.png", new File([new Uint8Array(8)], "wood.png")],
    ]);
});

describe("LocalMedia", () => {
    it("serves mounted files through fetch", async () => {
        const url = `${mount.baseUrl}Arcade/Arcade.fbx`;
        const res = await fetch(url);
        expect(res.status).toBe(200);
        expect([...new Uint8Array(await res.arrayBuffer())]).toEqual([1, 2, 3]);
        const head = await fetch(new URL(url, "http://localhost/"), { method: "HEAD" });
        expect(head.headers.get("content-length")).toBe("3");
        expect((await fetch(`${mount.baseUrl}Arcade/missing.fbx`)).status).toBe(404);
        expect(served).not.toHaveBeenCalled();
    });

    it("passes other URLs to the original fetch", async () => {
        expect(await (await fetch("/Falcor/media/test_scenes/cornell_box.pyscene")).text()).toBe("served");
        expect(served).toHaveBeenCalledTimes(1);
    });

    it("finds local paths behind an asset base and by URL", () => {
        expect(isLocalMediaUrl(`/web-falcor${mount.baseUrl}Arcade/Arcade.fbx`)).toBe(true);
        expect(getLocalFile(`http://localhost${mount.baseUrl}Arcade/Arcade.fbx`)?.name).toBe("Arcade.fbx");
        expect(isLocalMediaUrl("/Falcor/media/Arcade/Arcade.fbx")).toBe(false);
    });

    it("reads preloaded text files synchronously", () => {
        expect(new TextDecoder().decode(readLocalFileSync(`${mount.baseUrl}Arcade/Arcade.pyscene`)!)).toContain("importScene");
        expect(readLocalFileSync(`${mount.baseUrl}Arcade/Arcade.fbx`)).toBeUndefined(); // binary: not preloaded
        expect(readLocalFileSync("/Falcor/media/x.py")).toBeNull();
    });

    it("lists directories", async () => {
        expect(listLocalDirectory(`${mount.baseUrl}Arcade`)).toEqual({ files: ["Arcade.pyscene", "Arcade.fbx"], dirs: ["Textures"] });
        expect(await fetchDirectoryListing(`${mount.baseUrl}Arcade/Textures/`)).toEqual({ files: ["wood.png"], dirs: [] });
    });

    it("unmounts", async () => {
        unmountLocalFiles(mount.baseUrl);
        expect((await fetch(`${mount.baseUrl}Arcade/Arcade.fbx`)).status).toBe(404);
    });
});
