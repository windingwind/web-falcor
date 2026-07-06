/**
 * Render pass I/O reflection mirroring Falcor/RenderGraph/RenderPassReflection.h.
 */

import { ResourceFormat } from "../Core/API/Formats.js";
import { ResourceBindFlags, ResourceType } from "../Core/API/Types.js";

export enum FieldVisibility {
    Undefined = 0,
    Input = 1 << 0,
    Output = 1 << 1,
    Internal = 1 << 2,
}

export enum FieldFlags {
    None = 0,
    Optional = 1 << 0,
    Persistent = 1 << 1,
}

export class Field {
    resourceType: ResourceType = ResourceType.Texture2D;
    width = 0; // 0 => use graph default (output size)
    height = 0;
    depth = 0;
    sampleCount = 1;
    mipCount = 1;
    arraySize = 1;
    format_: ResourceFormat = ResourceFormat.Unknown;
    bindFlags_: ResourceBindFlags = ResourceBindFlags.None;
    flags_: FieldFlags = FieldFlags.None;
    visibility_: FieldVisibility = FieldVisibility.Undefined;

    constructor(
        public name_: string,
        public desc_: string,
        visibility: FieldVisibility,
    ) {
        this.visibility_ = visibility;
    }

    texture2D(width = 0, height = 0, sampleCount = 1, mipCount = 1, arraySize = 1): this {
        this.resourceType = ResourceType.Texture2D;
        this.width = width;
        this.height = height;
        this.sampleCount = sampleCount;
        this.mipCount = mipCount;
        this.arraySize = arraySize;
        return this;
    }
    texture3D(width = 0, height = 0, depth = 0): this {
        this.resourceType = ResourceType.Texture3D;
        this.width = width;
        this.height = height;
        this.depth = depth;
        return this;
    }
    format(f: ResourceFormat): this { this.format_ = f; return this; }
    bindFlags(flags: ResourceBindFlags): this { this.bindFlags_ = flags; return this; }
    flags(flags: FieldFlags): this { this.flags_ = flags; return this; }
    visibility(vis: FieldVisibility): this { this.visibility_ = vis; return this; }

    isOptional(): boolean { return (this.flags_ & FieldFlags.Optional) !== 0; }
    isInput(): boolean { return (this.visibility_ & FieldVisibility.Input) !== 0; }
    isOutput(): boolean { return (this.visibility_ & FieldVisibility.Output) !== 0; }
    isInternal(): boolean { return (this.visibility_ & FieldVisibility.Internal) !== 0; }

    /** Mirrors Field::merge: combines a connected output's field with an input's requirements. */
    merge(other: Field): this {
        if (this.format_ === ResourceFormat.Unknown) this.format_ = other.format_;
        this.bindFlags_ |= other.bindFlags_;
        return this;
    }
}

export class RenderPassReflection {
    readonly fields: Field[] = [];

    private add(name: string, desc: string, visibility: FieldVisibility): Field {
        const field = new Field(name, desc, visibility);
        this.fields.push(field);
        return field;
    }

    addInput(name: string, desc: string): Field {
        return this.add(name, desc, FieldVisibility.Input).bindFlags(ResourceBindFlags.ShaderResource);
    }
    addOutput(name: string, desc: string): Field {
        return this.add(name, desc, FieldVisibility.Output).bindFlags(ResourceBindFlags.UnorderedAccess | ResourceBindFlags.RenderTarget | ResourceBindFlags.ShaderResource);
    }
    addInternal(name: string, desc: string): Field {
        return this.add(name, desc, FieldVisibility.Internal);
    }
    addInputOutput(name: string, desc: string): Field {
        return this.add(name, desc, FieldVisibility.Input | FieldVisibility.Output);
    }

    getField(name: string): Field | undefined {
        return this.fields.find((f) => f.name_ === name);
    }

    /**
     * Clones a connected source field under this pass's input field name — used
     * by RenderGraph to build CompileData::connectedResources.
     */
    addConnectedField(name: string, src: Field): Field {
        const f = new Field(name, src.desc_, src.visibility_);
        f.resourceType = src.resourceType;
        f.width = src.width;
        f.height = src.height;
        f.depth = src.depth;
        f.sampleCount = src.sampleCount;
        f.mipCount = src.mipCount;
        f.arraySize = src.arraySize;
        f.format_ = src.format_;
        f.bindFlags_ = src.bindFlags_;
        f.flags_ = src.flags_;
        this.fields.push(f);
        return f;
    }
}
