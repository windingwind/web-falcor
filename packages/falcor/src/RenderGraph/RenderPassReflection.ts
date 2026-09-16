/**
 * Render pass I/O reflection mirroring Falcor/RenderGraph/RenderPassReflection.h.
 */

import { ResourceFormat } from "../Core/API/Formats.js";
import { ResourceBindFlags, ResourceType } from "../Core/API/Types.js";
import { kMaxPossible } from "../Core/API/Texture.js";
import { RuntimeError } from "../Core/Error.js";
import { Logger } from "../Utils/Logger.js";

export enum FieldVisibility {
    Undefined = 0,
    Input = 1 << 0,
    Output = 1 << 1,
    Internal = 1 << 2,
}

export enum FieldFlags {
    None = 0,
    Optional = 1 << 0,
    /** Resource must survive between execute() calls (web: also kept across recompiles while its description is unchanged). */
    Persistent = 1 << 1,
}

/** Mirrors RenderPassReflection::Field::Type. */
export enum FieldType {
    Texture1D,
    Texture2D,
    Texture3D,
    TextureCube,
    RawBuffer,
}

/** Mirrors Field::kMaxMipLevels: request the full mip chain. */
export const kMaxMipLevels = kMaxPossible;

/** Mirrors resourceTypeToFieldType (buffers map to RawBuffer on the web). */
export function resourceTypeToFieldType(type: ResourceType): FieldType {
    switch (type) {
        case ResourceType.Buffer: return FieldType.RawBuffer;
        case ResourceType.Texture1D: return FieldType.Texture1D;
        case ResourceType.Texture2D:
        case ResourceType.Texture2DMultisample: return FieldType.Texture2D;
        case ResourceType.Texture3D: return FieldType.Texture3D;
        case ResourceType.TextureCube: return FieldType.TextureCube;
        default: throw new RuntimeError(`resourceTypeToFieldType - no FieldType for ResourceType ${ResourceType[type]}`);
    }
}

/** Resource type a field allocates as. */
export function fieldTypeToResourceType(type: FieldType, sampleCount = 1): ResourceType {
    switch (type) {
        case FieldType.RawBuffer: return ResourceType.Buffer;
        case FieldType.Texture1D: return ResourceType.Texture1D;
        case FieldType.Texture2D: return sampleCount > 1 ? ResourceType.Texture2DMultisample : ResourceType.Texture2D;
        case FieldType.Texture3D: return ResourceType.Texture3D;
        case FieldType.TextureCube: return ResourceType.TextureCube;
    }
}

export class Field {
    type_: FieldType = FieldType.Texture2D;
    /** Texels for textures, bytes for raw buffers; 0 => graph default (output size). */
    width = 0;
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
        public name_ = "",
        public desc_ = "",
        visibility: FieldVisibility = FieldVisibility.Undefined,
    ) {
        this.visibility_ = visibility;
    }

    /** Mirrors Field::isValid (logs the reason, like native). */
    isValid(): boolean {
        if (this.sampleCount > 1 && this.mipCount > 1) {
            Logger.error(`Trying to create a multisampled RenderPassReflection::Field '${this.name_}' with mip-count larger than 1. This is illegal.`);
            return false;
        }
        if (this.isInternal() && this.isOptional()) {
            Logger.error("Internal resource can't be optional, since there will never be a graph edge that forces their creation");
            return false;
        }
        return true;
    }

    rawBuffer(size: number): this {
        this.type_ = FieldType.RawBuffer;
        this.width = size;
        this.height = this.depth = this.arraySize = this.mipCount = 0;
        return this;
    }
    texture1D(width = 0, mipCount = 1, arraySize = 1): this {
        this.type_ = FieldType.Texture1D;
        this.width = width;
        this.height = 1;
        this.depth = 1;
        this.sampleCount = 1;
        this.mipCount = mipCount;
        this.arraySize = arraySize;
        return this;
    }
    texture2D(width = 0, height = 0, sampleCount = 1, mipCount = 1, arraySize = 1): this {
        this.type_ = FieldType.Texture2D;
        this.width = width;
        this.height = height;
        this.depth = 1;
        this.sampleCount = sampleCount;
        this.mipCount = mipCount;
        this.arraySize = arraySize;
        return this;
    }
    texture3D(width = 0, height = 0, depth = 0, arraySize = 1): this {
        this.type_ = FieldType.Texture3D;
        this.width = width;
        this.height = height;
        this.depth = depth;
        this.sampleCount = 1;
        this.mipCount = 1;
        this.arraySize = arraySize;
        return this;
    }
    textureCube(width = 0, height = 0, mipCount = 1, arraySize = 1): this {
        this.type_ = FieldType.TextureCube;
        this.width = width;
        this.height = height;
        this.depth = 1;
        this.sampleCount = 1;
        this.mipCount = mipCount;
        this.arraySize = arraySize;
        return this;
    }

    /** Mirrors Field::resourceType: dispatches to the typed builder, warning on ignored dimensions. */
    resourceType(type: FieldType, width: number, height: number, depth: number, sampleCount: number, mipCount: number, arraySize: number): this {
        const warn = (msg: string) => Logger.warning(`RenderPassReflection::Field::resourceType - ${msg} for ${FieldType[type]}.`);
        switch (type) {
            case FieldType.RawBuffer:
                if (height > 0 || depth > 0 || sampleCount > 0) warn("height, depth, sampleCount must be 0");
                return this.rawBuffer(width);
            case FieldType.Texture1D:
                if (height > 1 || depth > 1 || sampleCount > 1) warn("height, depth, sampleCount must be either 0 or 1");
                return this.texture1D(width, mipCount, arraySize);
            case FieldType.Texture2D:
                if (depth > 1) warn("depth must be either 0 or 1");
                return this.texture2D(width, height, sampleCount, mipCount, arraySize);
            case FieldType.Texture3D:
                if (sampleCount > 1 || mipCount > 1) warn("sampleCount, mipCount must be either 0 or 1");
                return this.texture3D(width, height, depth, arraySize);
            case FieldType.TextureCube:
                if (depth > 1 || sampleCount > 1) warn("depth, sampleCount must be either 0 or 1");
                return this.textureCube(width, height, mipCount, arraySize);
            default:
                throw new RuntimeError(`RenderPassReflection::Field::resourceType - ${type} is not a valid Field type`);
        }
    }

    format(f: ResourceFormat): this { this.format_ = f; return this; }
    bindFlags(flags: ResourceBindFlags): this { this.bindFlags_ = flags; return this; }
    flags(flags: FieldFlags): this { this.flags_ = flags; return this; }
    visibility(vis: FieldVisibility): this { this.visibility_ = vis; return this; }
    name(name: string): this { this.name_ = name; return this; }
    desc(desc: string): this { this.desc_ = desc; return this; }

    /** Resource::Type this field allocates as. */
    getResourceType(): ResourceType { return fieldTypeToResourceType(this.type_, this.sampleCount); }

    isOptional(): boolean { return (this.flags_ & FieldFlags.Optional) !== 0; }
    isPersistent(): boolean { return (this.flags_ & FieldFlags.Persistent) !== 0; }
    isInput(): boolean { return (this.visibility_ & FieldVisibility.Input) !== 0; }
    isOutput(): boolean { return (this.visibility_ & FieldVisibility.Output) !== 0; }
    isInternal(): boolean { return (this.visibility_ & FieldVisibility.Internal) !== 0; }

    /**
     * Mirrors Field::merge: fills unspecified (0/Unknown) properties from `other`;
     * throws when both sides specify different values or the types differ.
     */
    merge(other: Field): this {
        const err = (msg: string): never => {
            throw new RuntimeError(`Can't merge RenderPassReflection::Fields. base(${this.name_}), newField(${other.name_}). ${msg}`);
        };
        if (this.type_ !== other.type_) err("mismatching types");
        if (other.isInternal() || this.isInternal()) err("internal fields can't be aliased");

        const mf = (key: "width" | "height" | "depth" | "arraySize" | "mipCount" | "sampleCount" | "format_", name: string) => {
            const none = key === "format_" ? ResourceFormat.Unknown : 0;
            if (other[key] !== none) {
                if (this[key] === none) (this as Record<typeof key, number>)[key] = other[key];
                else if (this[key] !== other[key]) err(`${name} already specified with a mismatching value in a different pass`);
            }
        };
        mf("width", "Width");
        mf("height", "Height");
        mf("depth", "Depth");
        mf("arraySize", "ArraySize");
        mf("mipCount", "MipCount");
        mf("sampleCount", "SampleCount");
        mf("format_", "Format");

        this.visibility_ |= other.visibility_;
        this.bindFlags_ |= other.bindFlags_;
        return this;
    }

    /** Mirrors Field::operator==. */
    equals(other: Field): boolean {
        return (
            this.type_ === other.type_ &&
            this.name_ === other.name_ &&
            this.desc_ === other.desc_ &&
            this.width === other.width &&
            this.height === other.height &&
            this.depth === other.depth &&
            this.sampleCount === other.sampleCount &&
            this.mipCount === other.mipCount &&
            this.arraySize === other.arraySize &&
            this.format_ === other.format_ &&
            this.bindFlags_ === other.bindFlags_ &&
            this.flags_ === other.flags_ &&
            this.visibility_ === other.visibility_
        );
    }

    clone(): Field {
        const f = new Field(this.name_, this.desc_, this.visibility_);
        f.type_ = this.type_;
        f.width = this.width;
        f.height = this.height;
        f.depth = this.depth;
        f.sampleCount = this.sampleCount;
        f.mipCount = this.mipCount;
        f.arraySize = this.arraySize;
        f.format_ = this.format_;
        f.bindFlags_ = this.bindFlags_;
        f.flags_ = this.flags_;
        return f;
    }
}

export class RenderPassReflection {
    readonly fields: Field[] = [];

    /** Mirrors RenderPassReflection::addField: same-named I/O fields merge their visibility. */
    addField(field: Field): Field {
        for (const existing of this.fields) {
            if (existing.name_ !== field.name_) continue;
            const io = FieldVisibility.Input | FieldVisibility.Output;
            const ioField = (existing.visibility_ & io) !== 0;
            const ioRequest = (field.visibility_ & io) !== 0;
            if (ioField && ioRequest) {
                existing.visibility_ |= field.visibility_;
            } else if ((existing.visibility_ & field.visibility_) !== field.visibility_) {
                Logger.warning(`Trying to add an existing field '${field.name_}' to RenderPassReflection, but the visibility flags mismatch. Overriding the previous definition`);
            }
            return existing;
        }
        this.fields.push(field);
        return field;
    }

    private add(name: string, desc: string, visibility: FieldVisibility): Field {
        return this.addField(new Field(name, desc, visibility));
    }

    /** Bind flags default to None and are resolved from the format at allocation (native ResourceCache rule). */
    addInput(name: string, desc: string): Field {
        return this.add(name, desc, FieldVisibility.Input);
    }
    addOutput(name: string, desc: string): Field {
        return this.add(name, desc, FieldVisibility.Output);
    }
    addInternal(name: string, desc: string): Field {
        return this.add(name, desc, FieldVisibility.Internal);
    }
    addInputOutput(name: string, desc: string): Field {
        return this.add(name, desc, FieldVisibility.Input | FieldVisibility.Output);
    }

    getFieldCount(): number {
        return this.fields.length;
    }

    getField(nameOrIndex: string | number): Field | undefined {
        if (typeof nameOrIndex === "number") return this.fields[nameOrIndex];
        return this.fields.find((f) => f.name_ === nameOrIndex);
    }

    /** Mirrors RenderPassReflection::operator== (order-independent field match). */
    equals(other: RenderPassReflection): boolean {
        if (other.fields.length !== this.fields.length) return false;
        return this.fields.every((f) => {
            const o = other.getField(f.name_);
            return o !== undefined && o.equals(f);
        });
    }

    /**
     * Clones a connected source field under this pass's input field name — used
     * by RenderGraph to build CompileData::connectedResources.
     */
    addConnectedField(name: string, src: Field): Field {
        const f = src.clone().name(name);
        this.fields.push(f);
        return f;
    }
}
