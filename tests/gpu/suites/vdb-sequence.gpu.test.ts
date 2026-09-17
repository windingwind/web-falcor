/**
 * `.vdb` grid sequences (GridVolume::loadGridSequence + updatePlayback): several
 * volumes feed one slot and the scene clock picks the frame with native's
 * formula — `(startFrame + floor(max(0, t) * frameRate)) % frameCount`.
 *
 * Only one grid can be bound at a time (WGSL has no binding arrays), so a frame
 * change re-uploads the grid buffer; both tests check the binding follows.
 *
 * Fetch the volumes with: npm run download:assets -- openvdb
 */

import { Grid, buildNanoVDBGrid, buildSphereGrid, initScripting, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq, SkipError } from "../harness/registry.js";

gpuTest("VdbSequence.proceduralFramesFollowTheClock", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    // A one-volume scene whose sequence is replaced with distinguishable frames.
    const scene = await runSceneScript(
        device,
        [
            "v = GridVolume('procedural')",
            "v.densityGrid = Grid.createSphere(1.0, 0.1)",
            "sceneBuilder.addGridVolume(v)",
            "camera = Camera()",
            "camera.position = float3(0, 0, 5)",
            "sceneBuilder.addCamera(camera)",
        ].join("\n"),
        "/Falcor/media",
    );
    const volume = scene.gridVolumes[0]!;

    // Spheres of growing radius: each frame has its own voxel count.
    const frames = [0.6, 0.8, 1.0, 1.2].map((radius) => new Grid(device, buildNanoVDBGrid(buildSphereGrid(radius, 0.1))));
    volume.setGridSequence("density", frames);
    volume.frameRate = 25;
    scene.finalizeGridVolumes(); // rebind after replacing the sequence
    const voxelCounts = frames.map((g) => g.voxelCount);
    console.error(`# procedural sequence: ${voxelCounts.join(", ")} active voxels per frame`);
    expectEq(new Set(voxelCounts).size, 4, "the frames are distinct");
    expectEq(volume.gridFrameCount, 4, "four frames");
    expectEq(scene.gridStats !== null, true, "a grid is bound");

    const expectedFrame = (t: number, start = 0) => (start + Math.floor(Math.max(0, t) * 25)) % 4;
    for (const t of [0, 0.02, 0.04, 0.08, 0.12, 0.16, 0.5, 1.37]) {
        scene.updateGridVolumePlayback(t);
        expectEq(volume.gridFrame, expectedFrame(t), `frame at t=${t}`);
        expectEq(volume.densityGrid!.voxelCount, voxelCounts[expectedFrame(t)]!, `bound grid at t=${t}`);
        // The scene must have re-uploaded the grid the frame selects.
        expectEq(scene.gridStats!.maxIndex.join(",") === frames[expectedFrame(t)]!.maxIndex.join(","), true, `uploaded grid at t=${t}`);
    }

    // Negative time clamps, playback can be switched off, startFrame offsets.
    scene.updateGridVolumePlayback(-3);
    expectEq(volume.gridFrame, 0, "negative time clamps to frame 0");
    volume.playbackEnabled = false;
    scene.updateGridVolumePlayback(1);
    expectEq(volume.gridFrame, 0, "playback disabled holds the frame");
    volume.playbackEnabled = true;
    volume.startFrame = 3;
    scene.updateGridVolumePlayback(0.04);
    expectEq(volume.gridFrame, expectedFrame(0.04, 3), "start frame offsets playback");
    volume.startFrame = 99;
    expectEq(volume.startFrame, 3, "start frame clamps to the last frame");
    // Frame rate is clamped like native's [1, 1000].
    volume.frameRate = 5000;
    expectEq(volume.frameRate, 1000, "frame rate clamps high");
    volume.frameRate = 0.1;
    expectEq(volume.frameRate, 1, "frame rate clamps low");
});

gpuTest("VdbSequence.loadsFromFiles", async ({ device }) => {
    if (!(await fetch("/Falcor/media/openvdb/smoke.vdb", { method: "HEAD" })).ok) {
        throw new SkipError("Falcor/media/openvdb missing (npm run download:assets -- openvdb)");
    }
    await initScripting("/node_modules/pyodide");
    const sceneSource = await (await fetch("/tests/oracle/assets/vdb-sequence.pyscene")).text();
    const scene = await runSceneScript(device, sceneSource, "/Falcor/media");

    const volume = scene.gridVolumes[0]!;
    expectEq(volume.gridFrameCount, 2, "two frames loaded from files");
    expectEq(volume.frameRate, 10, "frame rate from the scene");
    const sequence = volume.getGridSequence("density");
    expectEq(sequence[0] !== sequence[1], true, "each file produced its own grid");
    expectEq(sequence[0]!.voxelCount, 1049275, "the volume decoded (openvdb.org smoke)");

    // Frame selection swaps which grid object the volume reports.
    scene.updateGridVolumePlayback(0);
    expectEq(volume.densityGrid === sequence[0], true, "frame 0 bound at t=0");
    scene.updateGridVolumePlayback(0.1);
    expectEq(volume.densityGrid === sequence[1], true, "frame 1 bound at t=0.1");
    scene.updateGridVolumePlayback(0.2);
    expectEq(volume.densityGrid === sequence[0], true, "the sequence wraps");
    console.error(`# file sequence: ${volume.gridFrameCount} frames, ${sequence[0]!.voxelCount} voxels per frame`);
});
