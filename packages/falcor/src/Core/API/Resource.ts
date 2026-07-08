/**
 * Resource base class mirroring Falcor/Core/API/Resource.h.
 * Barrier/state tracking methods are intentionally absent: WebGPU tracks
 * resource states internally (documented divergence, docs §9).
 */

import type { Device } from "./Device.js";
import { ResourceBindFlags, ResourceType } from "./Types.js";

export abstract class Resource {
    protected constructor(
        public readonly device: Device,
        public readonly type: ResourceType,
        public readonly bindFlags: ResourceBindFlags,
    ) {}

    name = "";

    abstract destroy(): void;
}
