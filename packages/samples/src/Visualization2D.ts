/** Mirrors Samples/Visualization2D: two full-screen 2D demos (markers, voxel normals) driven by the mouse. */

import { FullScreenPass, MouseButton, MouseEventType, SampleApp, type Fbo, type MouseEvent, type RenderContext, type SampleAppConfig, type UIWidgets } from "@web-falcor/falcor";

export enum Visualization2DScene {
    MarkerDemo,
    VoxelNormals,
}

const kShaders: Record<Visualization2DScene, string> = {
    [Visualization2DScene.MarkerDemo]: "Samples/Visualization2D/Visualization2d.ps.slang",
    [Visualization2DScene.VoxelNormals]: "Samples/Visualization2D/VoxelNormals.ps.slang",
};
const kModeList = ["Marker demo", "Voxel normals"];

export class Visualization2D extends SampleApp {
    static readonly config: SampleAppConfig = { windowDesc: { title: "Falcor 2D Visualization", resizableWindow: true, width: 1400, height: 1000, enableVSync: true } };
    private mainPass: FullScreenPass | null = null;
    private leftButtonDown = false;
    mousePosition: [number, number] = [0.2, 0.1];
    readonly voxelNormalsGUI = { showNormalField: false, showBoxes: true, showBoxDiagonals: true, showBorderLines: false, showBoxAroundPoint: false };
    private selectedScene = Visualization2DScene.MarkerDemo;

    override onLoad(): void {
        this.createRenderPass();
    }

    /** Selects a demo (the Gui's "Scene selection"). */
    selectScene(scene: Visualization2DScene): void {
        this.selectedScene = scene;
        this.createRenderPass();
    }

    override onFrameRender(renderContext: RenderContext, targetFbo: Fbo): void {
        const v = this.mainPass!.getRootVar();
        const cb = v["Visual2DCB"];
        cb["iResolution"] = [targetFbo.width, targetFbo.height];
        cb["iGlobalTime"] = this.getGlobalClock().getTime();
        cb["iMousePosition"] = this.mousePosition;
        if (this.selectedScene === Visualization2DScene.VoxelNormals) {
            const g = this.voxelNormalsGUI;
            const n = v["VoxelNormalsCB"];
            n["iShowNormalField"] = g.showNormalField ? 1 : 0;
            n["iShowBoxes"] = g.showBoxes ? 1 : 0;
            n["iShowBoxDiagonals"] = g.showBoxDiagonals ? 1 : 0;
            n["iShowBorderLines"] = g.showBorderLines ? 1 : 0;
            n["iShowBoxAroundPoint"] = g.showBoxAroundPoint ? 1 : 0;
        }
        this.mainPass!.execute(renderContext, targetFbo);
    }

    override onGuiRender(gui: UIWidgets): void {
        const w = gui.group("Visualization 2D");
        w.dropdown("Scene selection", kModeList, kModeList[this.selectedScene]!, (v) => this.selectScene(kModeList.indexOf(v)));
        const clock = this.getGlobalClock();
        w.checkbox("Pause time", clock.isPaused(), (paused) => (paused ? clock.pause() : clock.play()));
        this.renderGlobalUI(w);
        if (this.selectedScene === Visualization2DScene.MarkerDemo) {
            w.text("Left-click and move mouse...");
        } else {
            const g = this.voxelNormalsGUI;
            w.text("Left-click and move mouse in the left boxes to display the normal there.");
            w.checkbox("Show normal field", g.showNormalField, (b) => (g.showNormalField = b));
            w.checkbox("Show boxes", g.showBoxes, (b) => (g.showBoxes = b));
            w.checkbox("Show box diagonals", g.showBoxDiagonals, (b) => (g.showBoxDiagonals = b));
            w.checkbox("Show border lines", g.showBorderLines, (b) => (g.showBorderLines = b));
            w.checkbox("Show box around point", g.showBoxAroundPoint, (b) => (g.showBoxAroundPoint = b));
        }
    }

    override onMouseEvent(mouseEvent: MouseEvent): boolean {
        switch (mouseEvent.type) {
            case MouseEventType.ButtonDown:
            case MouseEventType.ButtonUp:
                if (mouseEvent.button === MouseButton.Left) {
                    this.leftButtonDown = mouseEvent.type === MouseEventType.ButtonDown;
                    return true;
                }
                return false;
            case MouseEventType.Move:
                if (this.leftButtonDown) {
                    this.mousePosition = mouseEvent.screenPos;
                    return true;
                }
                return false;
            default:
                return false;
        }
    }

    private createRenderPass(): void {
        this.mainPass = FullScreenPass.create(this.getDevice(), { path: kShaders[this.selectedScene] });
    }
}
