// Scene panel mirroring Scene::renderUI (camera, lights, materials, env map,
// animation) on the DOM. Edits go through the runtime-edit API
// (camera setters, Scene.updateLights / updateMaterial) and restart accumulation.
import { LightType, Logger, MaterialType, float3, float4, type Scene } from "@web-falcor/falcor";

export interface ScenePanelHooks {
    /** Called after any edit (viewer restarts accumulation). */
    notify: () => void;
    /** Rebuilds the panel (lists that changed, e.g. viewpoints). */
    rebuild?: () => void;
    /** Viewer-level "Animate Scene" flag (mirrors AnimationController::setEnabled). */
    getAnimate: () => boolean;
    setAnimate: (v: boolean) => void;
    /** Camera controller selection (mirrors Scene::setCameraController / setUpDirection / setCameraSpeed). */
    cameraControl?: {
        types: readonly string[];
        getType: () => string;
        setType: (v: string) => void;
        upNames: readonly string[];
        getUp: () => number;
        setUp: (index: number) => void;
        getSpeed: () => number;
        setSpeed: (v: number) => void;
    };
}

const kLightTypeNames = ["Point", "Directional", "Distant", "Rect", "Disc", "Sphere"];

/** Small DOM helpers (numeric `var` controls native uses for unbounded values). */
class Dom {
    constructor(
        readonly root: HTMLElement,
        private readonly notify: () => void,
    ) {}
    private row(label: string): HTMLElement {
        const row = document.createElement("label");
        row.className = "ui-row";
        const span = document.createElement("span");
        span.textContent = label;
        row.appendChild(span);
        this.root.appendChild(row);
        return row;
    }
    text(label: string): void {
        const div = document.createElement("div");
        div.className = "ui-text";
        div.textContent = label;
        this.root.appendChild(div);
    }
    button(label: string, onClick: () => void): void {
        const btn = document.createElement("button");
        btn.textContent = label;
        btn.onclick = () => {
            onClick();
            this.notify();
        };
        this.root.appendChild(btn);
    }
    checkbox(label: string, value: boolean, onChange: (v: boolean) => void): void {
        const input = document.createElement("input");
        input.type = "checkbox";
        input.checked = value;
        input.onchange = () => {
            onChange(input.checked);
            this.notify();
        };
        this.row(label).appendChild(input);
    }
    /** Numeric input (native Gui::var). */
    num(label: string, value: number, step: number, onChange: (v: number) => void, min?: number, max?: number): void {
        this.row(label).appendChild(this.numInput(value, step, onChange, min, max));
    }
    /** N numeric inputs on one row (float3/float4 vars). */
    vec(label: string, values: number[], step: number, onChange: (i: number, v: number) => void, min?: number, max?: number): void {
        const row = this.row(label);
        row.classList.add("ui-vec");
        values.forEach((v, i) => row.appendChild(this.numInput(v, step, (x) => onChange(i, x), min, max)));
    }
    dropdown(label: string, options: readonly string[], value: string, onChange: (v: string) => void): void {
        const sel = document.createElement("select");
        for (const o of options) {
            const opt = document.createElement("option");
            opt.value = o;
            opt.textContent = o;
            opt.selected = o === value;
            sel.appendChild(opt);
        }
        sel.onchange = () => {
            onChange(sel.value);
            this.notify();
        };
        this.row(label).appendChild(sel);
    }
    group(label: string, open = false): Dom {
        const details = document.createElement("details");
        details.open = open;
        const summary = document.createElement("summary");
        summary.textContent = label;
        details.appendChild(summary);
        this.root.appendChild(details);
        return new Dom(details, this.notify);
    }
    private numInput(value: number, step: number, onChange: (v: number) => void, min?: number, max?: number): HTMLInputElement {
        const input = document.createElement("input");
        input.type = "number";
        input.step = String(step);
        if (min !== undefined) input.min = String(min);
        if (max !== undefined) input.max = String(max);
        input.value = String(Number(value.toFixed(6)));
        input.onchange = () => {
            const v = Number(input.value);
            if (!Number.isFinite(v)) return;
            onChange(min !== undefined && max !== undefined ? Math.min(max, Math.max(min, v)) : v);
            this.notify();
        };
        return input;
    }
}

function v3(v: { x: number; y: number; z: number }): number[] {
    return [v.x, v.y, v.z];
}

/** Builds the Scene section (returns false when there is no scene). */
export function buildScenePanel(container: HTMLElement, scene: Scene | null, hooks: ScenePanelHooks): boolean {
    if (!scene) return false;
    const ui = new Dom(container, hooks.notify);

    if (scene.isAnimated()) ui.checkbox("Animate Scene", hooks.getAnimate(), hooks.setAnimate);
    if (scene.hasAnimation()) ui.checkbox("Loop Animations", scene.isLooped(), (v) => scene.setIsLooped(v));
    if (hooks.cameraControl) {
        const cc = hooks.cameraControl;
        ui.dropdown("Up Direction", cc.upNames, cc.upNames[cc.getUp()] ?? cc.upNames[2]!, (v) => cc.setUp(Math.max(0, cc.upNames.indexOf(v))));
        ui.dropdown("Camera Controller", cc.types, cc.getType(), cc.setType);
        ui.num("Camera Speed", cc.getSpeed(), 0.01, cc.setSpeed, 0);
    }

    // Mirrors Scene::renderUI's camera selection and viewpoints (F3 adds one, as natively).
    {
        const names = scene.cameras.map((c, i) => `${i}: ${c.name}`);
        if (names.length > 1) {
            ui.dropdown("Selected Camera", names, names[scene.getSelectedCameraIndex()]!, (v) => {
                scene.selectCamera(Number(v.split(":")[0]));
                hooks.rebuild?.();
            });
        }
        ui.button("Add Viewpoint", () => {
            scene.addViewpoint();
            hooks.rebuild?.();
        });
        const count = scene.getViewpointCount();
        if (count > 1) {
            ui.button("Remove Viewpoint", () => {
                scene.removeViewpoint();
                hooks.rebuild?.();
            });
            let animationLength = 30;
            ui.num("Animation Length", animationLength, 1, (v) => (animationLength = Math.round(v)), 1, 120);
            // Native writes the file through a save dialog; the browser downloads it.
            ui.button("Save Viewpoints", () => {
                const a = document.createElement("a");
                a.href = URL.createObjectURL(new Blob([scene.getViewpointsScript(animationLength)], { type: "text/plain" }));
                a.download = "cameraPath.txt";
                a.click();
                URL.revokeObjectURL(a.href);
            });
            const views = Array.from({ length: count }, (_, i) => (i === 0 ? "Default Viewpoint" : `Viewpoint ${i}`));
            ui.dropdown("Viewpoints", views, views[scene.getCurrentViewpoint()]!, (v) => {
                scene.selectViewpoint(views.indexOf(v));
                hooks.rebuild?.();
            });
        }
    }

    // Camera (mirrors Camera::renderUI; shutter/ISO are not modelled on the web).
    {
        const cam = scene.camera;
        const g = ui.group("Camera", true);
        g.num("Focal Length", cam.getFocalLength(), 0.25, (v) => cam.setFocalLength(v), 0);
        g.num("Aspect Ratio", cam.getAspectRatio(), 0.001, (v) => cam.setAspectRatio(v), 0);
        g.num("Focal Distance", cam.getFocalDistance(), 0.05, (v) => cam.setFocalDistance(v), 0);
        g.num("Aperture Radius", cam.getApertureRadius(), 0.001, (v) => cam.setApertureRadius(v), 0);
        const data = cam.getData();
        g.vec("Depth Range", [data.nearZ, data.farZ], 0.1, (i, v) => {
            const d = cam.getData();
            cam.setDepthRange(i === 0 ? v : d.nearZ, i === 1 ? v : d.farZ);
        }, 0);
        const vecSetter = (get: () => float3, set: (v: float3) => void) => (i: number, v: number) => {
            const cur = get();
            const arr = v3(cur);
            arr[i] = v;
            set(new float3(arr[0]!, arr[1]!, arr[2]!));
        };
        g.vec("Position", v3(cam.getPosition()), 0.001, vecSetter(() => cam.getPosition(), (v) => cam.setPosition(v)));
        g.vec("Target", v3(cam.getTarget()), 0.001, vecSetter(() => cam.getTarget(), (v) => cam.setTarget(v)));
        g.vec("Up", v3(cam.getUpVector()), 0.001, vecSetter(() => cam.getUpVector(), (v) => cam.setUpVector(v)));
        g.button("Dump", () => {
            // Mirrors Camera::dumpProperties: the pyscene lines that reproduce this camera.
            const f = (v: float3) => `float3(${v.x}, ${v.y}, ${v.z})`;
            console.log(
                [
                    `camera.position = ${f(cam.getPosition())}`,
                    `camera.target = ${f(cam.getTarget())}`,
                    `camera.up = ${f(cam.getUpVector())}`,
                    `camera.focalLength = ${cam.getFocalLength()}`,
                    `camera.focalDistance = ${cam.getFocalDistance()}`,
                    `camera.apertureRadius = ${cam.getApertureRadius()}`,
                ].join("\n"),
            );
        });
    }

    // Env map (mirrors EnvMap::renderUI; no file dialog on the web).
    const env = scene.getEnvMap();
    if (env) {
        const g = ui.group("EnvMap");
        g.vec("Rotation XYZ", [...env.rotationDeg], 0.5, (i, v) => {
            const r: [number, number, number] = [...env.rotationDeg];
            r[i] = v;
            env.setRotation(r);
        }, -360, 360);
        g.num("Intensity", env.intensity, 0.01, (v) => (env.intensity = v), 0, 1000000);
        g.vec("Color tint", [...env.tint], 0.01, (i, v) => (env.tint[i] = v), 0, 1);
        g.text(`Resolution: ${env.texture.width}x${env.texture.height}, mips: ${env.texture.mipCount}`);
    }

    // Mirrors Scene::renderUI "Render Settings" (master light-usage switches; the graph recompiles passes).
    {
        const g = ui.group("Render Settings");
        const rs = scene.renderSettings;
        g.checkbox("Use environment light", rs.useEnvLight, (v) => (rs.useEnvLight = v));
        g.checkbox("Use analytic lights", rs.useAnalyticLights, (v) => (rs.useAnalyticLights = v));
        g.checkbox("Use emissive", rs.useEmissiveLights, (v) => (rs.useEmissiveLights = v));
        g.checkbox("Use grid volumes", rs.useGridVolumes, (v) => (rs.useGridVolumes = v));
    }

    // Lights (mirrors Light::renderUI + per-type controls). Color/intensity split like native getColorForUI/getIntensityForUI.
    {
        const g = ui.group("Lights");
        for (let i = 0; i < scene.getLightCount(); i++) {
            const light = scene.getLight(i);
            const type = kLightTypeNames[light.type] ?? "Light";
            const lg = g.group(`${i}: ${light.name ?? type} (${type})`);
            const scalar = () => Math.max(light.intensity.x, light.intensity.y, light.intensity.z);
            const color = () => {
                const s = scalar();
                return s > 0 ? [light.intensity.x / s, light.intensity.y / s, light.intensity.z / s] : [1, 1, 1];
            };
            const apply = () => scene.updateLights();
            lg.vec("Color", color(), 0.01, (ci, v) => {
                const c = color();
                c[ci] = v;
                const s = scalar();
                light.intensity = new float3(c[0]! * s, c[1]! * s, c[2]! * s);
                apply();
            }, 0, 1);
            lg.num("Intensity", scalar(), 0.1, (v) => {
                const c = color();
                light.intensity = new float3(c[0]! * v, c[1]! * v, c[2]! * v);
                apply();
            }, 0);
            if (light.posW && (light.type === LightType.Point)) {
                lg.vec("World Position", v3(light.posW), 0.01, (ci, v) => {
                    const p = v3(light.posW!);
                    p[ci] = v;
                    light.posW = new float3(p[0]!, p[1]!, p[2]!);
                    apply();
                });
            }
            if (light.dirW && (light.type === LightType.Point || light.type === LightType.Directional || light.type === LightType.Distant)) {
                lg.vec("Direction", v3(light.dirW), 0.01, (ci, v) => {
                    const d = v3(light.dirW!);
                    d[ci] = v;
                    const len = Math.hypot(d[0]!, d[1]!, d[2]!);
                    if (len > 0) light.dirW = new float3(d[0]! / len, d[1]! / len, d[2]! / len);
                    apply();
                }, -1, 1);
            }
            if (light.type === LightType.Point) {
                lg.num("Opening Angle", light.openingAngle ?? Math.PI, 0.01, (v) => {
                    light.openingAngle = Math.min(Math.PI, Math.max(0, v));
                    light.penumbraAngle = Math.min(light.penumbraAngle ?? 0, light.openingAngle);
                    apply();
                }, 0, Math.PI);
                lg.num("Penumbra Width", light.penumbraAngle ?? 0, 0.01, (v) => {
                    light.penumbraAngle = Math.min(light.openingAngle ?? Math.PI, Math.max(0, v));
                    apply();
                }, 0, Math.PI);
            }
        }
    }

    // Materials (mirrors MaterialSystem::renderUI: dropdown + selected material's BasicMaterial UI).
    if (scene.getMaterialCount() > 0) {
        const g = ui.group("Materials");
        const names: string[] = [];
        for (let i = 0; i < scene.getMaterialCount(); i++) names.push(`${i}: ${scene.getMaterial(i).name ?? "material"}`);
        const body = document.createElement("div");
        let selected = 0;
        const render = () => {
            body.textContent = "";
            const m = scene.getMaterial(selected);
            const mg = new Dom(body, hooks.notify);
            const apply = () => scene.updateMaterial(selected);
            mg.text(`Type: ${m.header?.materialType !== undefined ? MaterialType[m.header.materialType] : "Standard"}`);
            const bc = m.basic.baseColor ?? new float4(1, 1, 1, 1);
            mg.vec("Base color", [bc.x, bc.y, bc.z, bc.w], 0.01, (i, v) => {
                const c = [bc.x, bc.y, bc.z, bc.w];
                c[i] = v;
                m.basic.baseColor = new float4(c[0]!, c[1]!, c[2]!, c[3]!);
                apply();
            }, 0, 1);
            const sp = m.basic.specular ?? new float4(0, 0, 0, 0);
            mg.vec("Specular params", [sp.x, sp.y, sp.z, sp.w], 0.01, (i, v) => {
                const c = [sp.x, sp.y, sp.z, sp.w];
                c[i] = v;
                m.basic.specular = new float4(c[0]!, c[1]!, c[2]!, c[3]!);
                apply();
            }, 0, 1);
            if (m.header?.emissive) {
                const em = m.basic.emissive ?? new float3(0, 0, 0);
                mg.vec("Emissive color", v3(em), 0.01, (i, v) => {
                    const c = v3(em);
                    c[i] = v;
                    m.basic.emissive = new float3(c[0]!, c[1]!, c[2]!);
                    apply();
                }, 0);
                mg.num("Emissive factor", m.basic.emissiveFactor ?? 1, 0.1, (v) => {
                    m.basic.emissiveFactor = v;
                    apply();
                }, 0);
            }
            mg.num("Index of refraction", m.header?.ior ?? 1.5, 0.01, (v) => {
                m.header = { ...(m.header ?? { materialType: MaterialType.Standard }), ior: v };
                apply();
            }, 1);
            mg.num("Specular transmission", m.basic.specularTransmission ?? 0, 0.01, (v) => {
                m.basic.specularTransmission = v;
                apply();
            }, 0, 1);
            mg.num("Diffuse transmission", m.basic.diffuseTransmission ?? 0, 0.01, (v) => {
                m.basic.diffuseTransmission = v;
                apply();
            }, 0, 1);
            mg.checkbox("Double sided", m.header?.doubleSided ?? false, (v) => {
                m.header = { ...(m.header ?? { materialType: MaterialType.Standard }), doubleSided: v };
                apply();
            });
            mg.checkbox("Thin surface", m.header?.thinSurface ?? false, (v) => {
                m.header = { ...(m.header ?? { materialType: MaterialType.Standard }), thinSurface: v };
                apply();
            });
        };
        g.dropdown("Material", names, names[0]!, (v) => {
            selected = Number(v.split(":")[0]);
            render();
        });
        g.root.appendChild(body);
        render();
    }

    // Mirrors Scene::renderUI "Statistics"; the text is refreshed whenever the group opens.
    {
        const g = ui.group("Statistics");
        // Not g.button: printing must not restart accumulation.
        const print = document.createElement("button");
        print.textContent = "Print to log";
        print.onclick = () => Logger.info("\n" + scene.getSceneStatsText());
        g.root.appendChild(print);
        const pre = document.createElement("pre");
        pre.className = "scene-stats";
        pre.style.margin = "4px 0";
        g.root.appendChild(pre);
        const details = g.root as HTMLDetailsElement;
        details.addEventListener("toggle", () => {
            if (details.open) pre.textContent = scene.getSceneStatsText();
        });
    }
    return true;
}
