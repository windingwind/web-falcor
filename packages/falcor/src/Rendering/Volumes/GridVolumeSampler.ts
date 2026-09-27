/**
 * Mirrors Rendering/Volumes/GridVolumeSampler.h: transmittance evaluation and distance sampling in grid volumes
 * (the shader side is the unmodified GridVolumeSampler.slang); the local-majorant modes need the BrickedGrid.
 */

import { defineEnumInfo, enumToString, stringToEnum } from "../../Core/Enum.js";
import type { DefineList } from "../../Core/Program/DefineList.js";
import type { UIWidgets } from "../../RenderGraph/UIWidgets.js";
import type { Scene } from "../../Scene/Scene.js";

/** GridVolumeSamplerParams.slang's TransmittanceEstimator. */
export enum TransmittanceEstimator {
    DeltaTracking,
    RatioTracking,
    /** BrickedGrid only. */
    RatioTrackingLocalMajorant,
}
export const kTransmittanceEstimatorInfo = defineEnumInfo([
    [TransmittanceEstimator.DeltaTracking, "DeltaTracking"],
    [TransmittanceEstimator.RatioTracking, "RatioTracking"],
    [TransmittanceEstimator.RatioTrackingLocalMajorant, "RatioTrackingLocalMajorant"],
]);

/** GridVolumeSamplerParams.slang's DistanceSampler. */
export enum DistanceSampler {
    DeltaTracking,
    /** BrickedGrid only. */
    DeltaTrackingLocalMajorant,
}
export const kDistanceSamplerInfo = defineEnumInfo([
    [DistanceSampler.DeltaTracking, "DeltaTracking"],
    [DistanceSampler.DeltaTrackingLocalMajorant, "DeltaTrackingLocalMajorant"],
]);

export function requiresBrickedGrid(mode: { transmittance: TransmittanceEstimator } | { distance: DistanceSampler }): boolean {
    return "transmittance" in mode ? mode.transmittance === TransmittanceEstimator.RatioTrackingLocalMajorant : mode.distance === DistanceSampler.DeltaTrackingLocalMajorant;
}

export interface GridVolumeSamplerOptions {
    transmittanceEstimator: TransmittanceEstimator;
    distanceSampler: DistanceSampler;
    useBrickedGrid: boolean;
}

export const kDefaultGridVolumeSamplerOptions: Readonly<GridVolumeSamplerOptions> = {
    transmittanceEstimator: TransmittanceEstimator.RatioTrackingLocalMajorant,
    distanceSampler: DistanceSampler.DeltaTrackingLocalMajorant,
    useBrickedGrid: true,
};

export class GridVolumeSampler {
    private options: GridVolumeSamplerOptions;

    constructor(
        readonly scene: Scene,
        options: Partial<GridVolumeSamplerOptions> = {},
    ) {
        this.options = { ...kDefaultGridVolumeSamplerOptions, ...options };
    }

    /** The shader defines for using the sampler. */
    getDefines(): Record<string, string> {
        return {
            GRID_VOLUME_SAMPLER_USE_BRICKEDGRID: String(+this.options.useBrickedGrid),
            GRID_VOLUME_SAMPLER_TRANSMITTANCE_ESTIMATOR: String(this.options.transmittanceEstimator),
            GRID_VOLUME_SAMPLER_DISTANCE_SAMPLER: String(this.options.distanceSampler),
        };
    }

    /** Adds the defines to a DefineList (native getDefines returns one). */
    addDefines(defines: DefineList): void {
        for (const [k, v] of Object.entries(this.getDefines())) defines.add(k, v);
    }

    /** Native binds nothing (the sampler reads the scene's volumes). */
    bindShaderData(_var: unknown): void {}

    /** Mirrors renderUI: modes needing the bricked grid switch it on, and turning it off leaves those modes. */
    renderUI(ui: UIWidgets, onChange: () => void = () => {}): void {
        const o = this.options;
        ui.checkbox("Use BrickedGrid", o.useBrickedGrid, (v) => {
            o.useBrickedGrid = v;
            if (!v) {
                if (requiresBrickedGrid({ transmittance: o.transmittanceEstimator })) o.transmittanceEstimator = TransmittanceEstimator.RatioTracking;
                if (requiresBrickedGrid({ distance: o.distanceSampler })) o.distanceSampler = DistanceSampler.DeltaTracking;
            }
            onChange();
        });
        ui.dropdown("Transmittance Estimator", kTransmittanceEstimatorInfo.map(([, n]) => n), enumToString(kTransmittanceEstimatorInfo, o.transmittanceEstimator), (v) => {
            o.transmittanceEstimator = stringToEnum(kTransmittanceEstimatorInfo, v);
            if (requiresBrickedGrid({ transmittance: o.transmittanceEstimator })) o.useBrickedGrid = true;
            onChange();
        });
        ui.dropdown("Distance Sampler", kDistanceSamplerInfo.map(([, n]) => n), enumToString(kDistanceSamplerInfo, o.distanceSampler), (v) => {
            o.distanceSampler = stringToEnum(kDistanceSamplerInfo, v);
            if (requiresBrickedGrid({ distance: o.distanceSampler })) o.useBrickedGrid = true;
            onChange();
        });
    }

    getOptions(): Readonly<GridVolumeSamplerOptions> {
        return this.options;
    }

    setOptions(options: GridVolumeSamplerOptions): void {
        this.options = { ...options };
    }
}
