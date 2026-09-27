/**
 * Mirrors Falcor/Core/Plugin.h: per plugin base class (any key object), the registered classes by type name.
 * A plugin class has static kPluginType/kPluginInfo and create (FALCOR_PLUGIN_CLASS).
 */

/** A registrable plugin class: FALCOR_PLUGIN_CLASS's type name and info plus the base class's create signature. */
export interface PluginClass<TInfo = unknown, TCreate extends (...args: never[]) => unknown = (...args: never[]) => unknown> {
    readonly kPluginType: string;
    readonly kPluginInfo: TInfo;
    readonly create: TCreate;
}

/** Instances of plugin classes report their class's type and info (FALCOR_PLUGIN_CLASS's getPluginType/Info). */
export abstract class PluginObject {
    getPluginType(): string {
        return (this.constructor as unknown as PluginClass).kPluginType;
    }
    getPluginInfo(): unknown {
        return (this.constructor as unknown as PluginClass).kPluginInfo;
    }
}

interface ClassDesc {
    type: string;
    info: unknown;
    create: (...args: never[]) => unknown;
    libraryId: number;
}

export class PluginManager {
    private static sInstance: PluginManager | undefined;
    private readonly classes = new Map<object, Map<string, ClassDesc>>();

    /** The global plugin manager (native PluginManager::instance()). */
    static instance(): PluginManager {
        return (PluginManager.sInstance ??= new PluginManager());
    }

    /** Registers a class under `base`; a type registered again is replaced (a reloaded library). */
    registerClass<TInfo, TCreate extends (...args: never[]) => unknown>(base: object, type: string, info: TInfo, create: TCreate, libraryId = 0): void {
        let byType = this.classes.get(base);
        if (!byType) this.classes.set(base, (byType = new Map()));
        byType.set(type, { type, info, create, libraryId });
    }

    hasClass(base: object, type: string): boolean {
        return this.classes.get(base)?.has(type) ?? false;
    }

    /** [type, info] of every class registered under `base`, in registration order. */
    getInfos<TInfo = unknown>(base: object): [string, TInfo][] {
        return [...(this.classes.get(base)?.values() ?? [])].map((d) => [d.type, d.info as TInfo]);
    }

    /** Creates a registered class through its create function; null for an unknown type, as natively. */
    createClass<TCreate extends (...args: never[]) => unknown>(base: object, type: string, ...args: Parameters<TCreate>): ReturnType<TCreate> | null {
        const desc = this.classes.get(base)?.get(type);
        return desc ? (desc.create as TCreate)(...args) as ReturnType<TCreate> : null;
    }

    /** Drops the classes a library registered (native unloadPlugin). */
    unregisterLibrary(libraryId: number): void {
        for (const byType of this.classes.values()) for (const [type, d] of byType) if (d.libraryId === libraryId) byType.delete(type);
    }
}

/** Mirrors PluginRegistry: what a plugin library's registerPlugin() receives to register its classes. */
export class PluginRegistry {
    constructor(
        private readonly pluginManager: PluginManager,
        private readonly libraryId: number,
    ) {}

    registerClass(base: object, cls: PluginClass): void {
        this.pluginManager.registerClass(base, cls.kPluginType, cls.kPluginInfo, cls.create, this.libraryId);
    }
}
