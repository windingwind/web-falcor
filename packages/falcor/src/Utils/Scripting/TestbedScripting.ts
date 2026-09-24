/**
 * Runs Falcor's Python scripts that drive a `falcor.Testbed` (scripts/python: balls,
 * test_replace_material, ...) unmodified. The script sees a `falcor` module with native
 * names and keyword arguments: Testbed, Device (create_buffer/_typed_buffer/
 * _structured_buffer/_texture), Buffer/Texture from_numpy/to_numpy, ComputePass
 * (file=/cs_entry=/defines=, `globals`, execute), the profiler's `event` context,
 * RenderGraph (create_pass/add_edge/mark_output), the resource enums and Logger.
 *
 * §9: shader and Python files next to the script are fetched into Pyodide's file
 * system; the script runs through runPythonAsync so frame() and to_numpy() can wait on
 * the GPU and the browser (pyodide.ffi.run_sync, JSPI). numpy comes from
 * tools/pyodide-packages (scripts/setup-web.mjs).
 */

import type { Device } from "../../Core/API/Device.js";
import type { Buffer } from "../../Core/API/Buffer.js";
import type { Texture } from "../../Core/API/Texture.js";
import { Texture as TextureClass, kMaxPossible } from "../../Core/API/Texture.js";
import { ResourceBindFlags, MemoryType, ResourceType } from "../../Core/API/Types.js";
import { FormatType, ResourceFormat, getFormatChannelCount, getFormatType, getNumChannelBits } from "../../Core/API/Formats.js";
import { ComputePass } from "../../Core/Pass/ComputePass.js";
import { StandaloneParameterBlock } from "../../Core/Program/ParameterBlock.js";
import { RenderGraph } from "../../RenderGraph/RenderGraph.js";
import { createPass } from "../../RenderGraph/RenderPass.js";
import { Properties } from "../Properties.js";
import { Logger } from "../Logger.js";
import { RuntimeError } from "../../Core/Error.js";
import { getPyodide } from "./Scripting.js";
import { Testbed, type TestbedOptions } from "./Testbed.js";

interface Pyodide {
    registerJsModule(name: string, module: object): void;
    runPythonAsync(code: string): Promise<unknown>;
    loadPackage(name: string | string[]): Promise<void>;
    FS: { mkdirTree(path: string): void; writeFile(path: string, data: string | Uint8Array): void; readFile(path: string, opts: { encoding: "utf8" }): string };
    toPy(v: unknown): unknown;
}

const toJs = (v: unknown): unknown => {
    const p = v as { toJs?: (o: object) => unknown };
    return p && typeof p.toJs === "function" ? p.toJs({ dict_converter: Object.fromEntries, create_proxies: false }) : v;
};

/** numpy dtype for a format's channels (native resourceFormatToDtype); null when there is none. */
function dtypeOf(format: ResourceFormat): string | null {
    const type = getFormatType(format);
    const bits = getNumChannelBits(format, 0);
    if (type === FormatType.Float) return bits === 16 ? "float16" : bits === 32 ? "float32" : null;
    const unsigned = type === FormatType.Uint || type === FormatType.Unorm || type === FormatType.UnormSrgb;
    const signed = type === FormatType.Sint || type === FormatType.Snorm;
    if (!unsigned && !signed) return null;
    return bits === 8 || bits === 16 || bits === 32 ? `${unsigned ? "uint" : "int"}${bits}` : null;
}

/** JS primitives the Python `falcor` module wraps. */
function makeJsModule(device: Device, testbedOptions: TestbedOptions, fsRead: (path: string) => string, created: Testbed[]) {
    return {
        createTestbed: (width: number, height: number, createWindow: boolean, title: string, showFPS: boolean) => {
            const t = new Testbed(device, { ...testbedOptions, width, height, createWindow, title, showFPS });
            created.push(t);
            return t;
        },
        createBuffer: (size: number, bindFlags: number, memoryType: number) => device.createBufferFromDesc({ size, bindFlags, memoryType }),
        createTypedBuffer: (format: number, count: number, bindFlags: number, memoryType: number) =>
            device.createBufferFromDesc({ size: count * Math.max(4, getFormatChannelCount(format) * (getNumChannelBits(format, 0) / 8)), format, bindFlags, memoryType }),
        createStructuredBuffer: (structSize: number, count: number, bindFlags: number, memoryType: number, createCounter: boolean) =>
            device.createBufferFromDesc({ size: structSize * count, structSize, bindFlags, memoryType, createCounter }),
        createTexture: (width: number, height: number, depth: number, format: number, arraySize: number, mipLevels: number, bindFlags: number) =>
            new TextureClass(device, {
                type: depth > 0 ? ResourceType.Texture3D : height > 0 ? ResourceType.Texture2D : ResourceType.Texture1D,
                width,
                height: Math.max(height, 1),
                depth: Math.max(depth, 1),
                format,
                arraySize,
                mipLevels: mipLevels < 0 ? kMaxPossible : mipLevels,
                bindFlags,
            }),
        bufferInfo: (b: Buffer) => ({ size: b.size, elementCount: b.elementCount, structSize: b.structSize, format: b.format, dtype: b.format !== undefined ? dtypeOf(b.format) : null, channels: b.format !== undefined ? getFormatChannelCount(b.format) : 1 }),
        bufferFromBytes: (b: Buffer, bytes: Uint8Array) => {
            if (bytes.byteLength > b.size) throw new RuntimeError(`numpy array is larger than the buffer (${bytes.byteLength} > ${b.size})`);
            b.setBlob(bytes, 0);
        },
        bufferToBytes: (b: Buffer) => b.getBlob(),
        textureInfo: (t: Texture, mip: number) => ({
            width: Math.max(1, t.width >> mip),
            height: Math.max(1, t.height >> mip),
            depth: t.type === ResourceType.Texture3D ? Math.max(1, t.depth >> mip) : 1,
            is3D: t.type === ResourceType.Texture3D,
            is1D: t.type === ResourceType.Texture1D,
            channels: getFormatChannelCount(t.format),
            dtype: dtypeOf(t.format),
            mipCount: t.mipCount,
            arraySize: t.arraySize,
        }),
        textureFromBytes: (t: Texture, bytes: Uint8Array, mip: number, slice: number) => t.setSubresourceBlob(mip, slice, bytes),
        textureToBytes: (t: Texture, mip: number, slice: number) => device.renderContext.readTextureSubresource(t, mip, slice),
        createComputePass: (file: string | null, csEntry: string, defines: unknown) => {
            const desc = { csEntry, defines: (toJs(defines) as Record<string, string | number>) ?? {} };
            if (!file) throw new RuntimeError("ComputePass: 'file' is required");
            // Script shaders live in Pyodide's FS; registry paths (Falcor shaders) otherwise.
            try {
                const code = fsRead(file);
                return ComputePass.create(device, { ...desc, modules: [{ name: file.slice(file.lastIndexOf("/") + 1).replace(/\.slang$/, ""), sources: [{ string: code, path: file }] }] });
            } catch (e) {
                if (!(e instanceof Error) || !/No such file|ENOENT|errno 44/i.test(String((e as { message?: string }).message) + String(e))) throw e;
                return ComputePass.create(device, { ...desc, path: file });
            }
        },
        setVar: (pass: ComputePass, path: unknown, value: unknown) => {
            const keys = toJs(path) as (string | number)[];
            let v = pass.getRootVar();
            for (const k of keys.slice(0, -1)) v = v[String(k)];
            v[String(keys[keys.length - 1])] = value instanceof StandaloneParameterBlock ? value : toJs(value);
        },
        getVar: (pass: ComputePass, path: unknown) => {
            const keys = toJs(path) as (string | number)[];
            let v = pass.getRootVar();
            for (const k of keys.slice(0, -1)) v = v[String(k)];
            return v[String(keys[keys.length - 1])].getResource();
        },
        execute: (pass: ComputePass, x: number, y: number, z: number) => pass.execute(device.renderContext, x, y, z),
        profilerBegin: (name: string) => device.profilerHook?.startEvent(name),
        profilerEnd: (name: string) => device.profilerHook?.endEvent(name),
        createRenderGraph: (t: Testbed, name: string) => t.createRenderGraph(name),
        graphCreatePass: (g: RenderGraph, name: string, type: string, props: unknown) => g.addPass(createPass(device, type, new Properties((toJs(props) as Record<string, never>) ?? {})), name),
        setLogVerbosity: (level: number) => (Logger.level = level),
    };
}

/** The Python side: native names, keyword arguments, numpy conversions. */
const kFalcorPython = String.raw`
import sys, types, enum
from pyodide.ffi import run_sync, to_js
# registerJsModule doesn't replace an imported module: drop the previous script's first.
sys.modules.pop("_falcor_testbed_js", None)
import _falcor_testbed_js as _js

falcor = types.ModuleType("falcor")

def _enum(name, members, flag):
    base = enum.IntFlag if flag else enum.IntEnum
    return base(name, members)

ResourceBindFlags = _enum("ResourceBindFlags", _BIND_FLAGS, True)
ResourceFormat = _enum("ResourceFormat", _FORMATS, False)
MemoryType = _enum("MemoryType", _MEMORY_TYPES, False)

class DeviceType(enum.IntEnum):
    Default = 0
    D3D12 = 1
    Vulkan = 2

class _LoggerLevel(enum.IntEnum):
    Disabled = 0
    Fatal = 1
    Error = 2
    Warning = 3
    Info = 4
    Debug = 5

class _LoggerMeta(type):
    Level = _LoggerLevel
    _verbosity = _LoggerLevel.Info
    @property
    def verbosity(cls):
        return cls._verbosity
    @verbosity.setter
    def verbosity(cls, v):
        cls._verbosity = v
        _js.setLogVerbosity(int(v))

class Logger(metaclass=_LoggerMeta):
    Level = _LoggerLevel

def _np():
    import numpy
    return numpy

def _unwrap(v):
    if isinstance(v, (Buffer, Texture)):
        return v._o
    if hasattr(v, "tolist") and not isinstance(v, (bool, int, float)):
        return to_js(v.tolist())
    if isinstance(v, (tuple, list)):
        return to_js(list(v))
    return v

class Buffer:
    def __init__(self, o):
        self._o = o
    @property
    def size(self): return int(_js.bufferInfo(self._o).size)
    @property
    def element_count(self): return int(_js.bufferInfo(self._o).elementCount)
    @property
    def struct_size(self): return int(_js.bufferInfo(self._o).structSize or 0)
    def from_numpy(self, data):
        np = _np()
        data = np.ascontiguousarray(data)
        _js.bufferFromBytes(self._o, to_js(memoryview(data.tobytes())))
    def to_numpy(self):
        np = _np()
        info = _js.bufferInfo(self._o)
        raw = np.frombuffer(bytes(run_sync(_js.bufferToBytes(self._o)).to_py()), dtype=np.uint8)
        if info.dtype:
            a = raw.view(info.dtype)
            return a if info.channels == 1 else a.reshape(-1, info.channels)
        return raw

class Texture:
    def __init__(self, o):
        self._o = o
    @property
    def width(self): return int(self._o.width)
    @property
    def height(self): return int(self._o.height)
    @property
    def depth(self): return int(self._o.depth)
    @property
    def mip_count(self): return int(self._o.mipCount)
    def from_numpy(self, data, mip_level=0, array_slice=0):
        np = _np()
        data = np.ascontiguousarray(data)
        _js.textureFromBytes(self._o, to_js(memoryview(data.tobytes())), mip_level, array_slice)
    def to_numpy(self, mip_level=0, array_slice=0):
        np = _np()
        info = _js.textureInfo(self._o, mip_level)
        raw = np.frombuffer(bytes(run_sync(_js.textureToBytes(self._o, mip_level, array_slice)).to_py()), dtype=np.uint8)
        if not info.dtype:
            return raw
        shape = []
        if info.depth > 1: shape.append(info.depth)
        if info.height > 1: shape.append(info.height)
        shape.append(info.width)
        if info.channels > 1: shape.append(info.channels)
        return raw.view(info.dtype).reshape(shape)

class Device:
    def __init__(self, type=DeviceType.Default, gpu=0, enable_debug_layer=False, enable_aftermath=False, _o=None):
        self._o = _o if _o is not None else _js_device
    def create_buffer(self, size, bind_flags=ResourceBindFlags(0), memory_type=MemoryType.DeviceLocal):
        return Buffer(_js.createBuffer(size, int(bind_flags), int(memory_type)))
    def create_typed_buffer(self, format, element_count, bind_flags=ResourceBindFlags(0), memory_type=MemoryType.DeviceLocal):
        return Buffer(_js.createTypedBuffer(int(format), element_count, int(bind_flags), int(memory_type)))
    def create_structured_buffer(self, struct_size, element_count, bind_flags=ResourceBindFlags(0), memory_type=MemoryType.DeviceLocal, create_counter=False):
        return Buffer(_js.createStructuredBuffer(struct_size, element_count, int(bind_flags), int(memory_type), create_counter))
    def create_texture(self, width, height=0, depth=0, format=ResourceFormat.Unknown, array_size=1, mip_levels=-1, bind_flags=ResourceBindFlags(0)):
        return Texture(_js.createTexture(width, height, depth, int(format), array_size, mip_levels, int(bind_flags)))
    @property
    def profiler(self):
        return _profiler

class _ProfilerEvent:
    def __init__(self, name): self.name = name
    def __enter__(self): _js.profilerBegin(self.name); return self
    def __exit__(self, *a): _js.profilerEnd(self.name); return False

class Profiler:
    enabled = False
    def event(self, name): return _ProfilerEvent(name)

_profiler = Profiler()

class _Vars:
    def __init__(self, pass_, path):
        object.__setattr__(self, "_p", pass_)
        object.__setattr__(self, "_path", path)
    def __getattr__(self, k): return _Vars(self._p, self._path + [k])
    def __getitem__(self, k): return _Vars(self._p, self._path + [k])
    def __setattr__(self, k, v): _js.setVar(self._p, to_js(self._path + [k]), _unwrap(v))
    def __setitem__(self, k, v): _js.setVar(self._p, to_js(self._path + [k]), _unwrap(v))

class ComputePass:
    def __init__(self, device, desc=None, defines=None, *, file=None, cs_entry="main", **kwargs):
        self._o = _js.createComputePass(str(file) if file is not None else None, cs_entry, to_js(dict(defines or {})))
    @property
    def globals(self): return _Vars(self._o, [])
    @property
    def root_var(self): return self.globals
    def execute(self, threads_x, threads_y=1, threads_z=1, compute_context=None):
        _js.execute(self._o, threads_x, threads_y, threads_z)

class RenderGraph:
    def __init__(self, o): self._o = o
    @property
    def name(self): return self._o.name
    def create_pass(self, name, type, dict={}):
        _js.graphCreatePass(self._o, name, type, to_js(dict))
    def add_edge(self, src, dst): self._o.addEdge(src, dst)
    def remove_edge(self, src, dst): self._o.removeEdge(src, dst)
    def mark_output(self, name, mask=7): self._o.markOutput(name, mask)
    def unmark_output(self, name): self._o.unmarkOutput(name)
    def get_pass(self, name): return self._o.getPass(name)

class Testbed:
    def __init__(self, width=1920, height=1080, create_window=False, device_type=DeviceType.Default, gpu=0, enable_debug_layers=False, enable_aftermath=False, title="Falcor Sample", show_fps=True, device=None):
        self._o = _js.createTestbed(width, height, create_window, title, show_fps)
        self._device = device if device is not None else Device()
        self._graph = None
    @property
    def device(self): return self._device
    @property
    def profiler(self): return _profiler
    @property
    def scene(self): return self._o.scene
    @property
    def clock(self): return self._o.clock
    @property
    def should_close(self): return bool(self._o.shouldClose)
    @property
    def show_ui(self): return bool(self._o.showUI)
    @show_ui.setter
    def show_ui(self, v): self._o.showUI = bool(v)
    @property
    def render_texture(self): return self._rt if hasattr(self, "_rt") else None
    @render_texture.setter
    def render_texture(self, t):
        self._rt = t
        self._o.renderTexture = t._o if t is not None else None
    @property
    def render_graph(self): return self._graph
    @render_graph.setter
    def render_graph(self, g):
        self._graph = g
        self._o.setRenderGraph(g._o if g is not None else None)
    def create_render_graph(self, name=""): return RenderGraph(_js.createRenderGraph(self._o, name))
    def run(self):
        while not self.should_close: self.frame()
    def resize_frame_buffer(self, width, height): self._o.resizeFrameBuffer(width, height)
    def load_scene(self, path, build_flags=0): run_sync(self._o.loadScene(str(path), int(build_flags)))
    def capture_output(self, path, output_index=0): run_sync(self._o.captureOutput(str(path), output_index))
    @property
    def screen(self):
        if not hasattr(self, "_screen"): self._screen = ui.Screen(self._o.screen)
        return self._screen
    def frame(self):
        run_sync(self._o.frame())
        if hasattr(self, "_screen"): self._screen._drain()
        _fire_timers()

for _n, _v in list(globals().items()):
    if _n[:1].isupper() and not _n.startswith("_"):
        setattr(falcor, _n, _v)
sys.modules["falcor"] = falcor

def _vector(name, n, scalar):
    fields = "xyzw"[:n]
    def new(cls, *args):
        if len(args) == 1 and hasattr(args[0], "__iter__"): args = tuple(args[0])
        if len(args) == 1: args = args * n
        if len(args) != n: raise TypeError(f"{name} takes 1 or {n} values")
        return tuple.__new__(cls, (scalar(a) for a in args))
    attrs = {"__new__": new, "__repr__": lambda s: f"{name}({', '.join(repr(v) for v in s)})"}
    for i, f in enumerate(fields): attrs[f] = property(lambda s, i=i: s[i])
    return type(name, (tuple,), attrs)

for _n in (2, 3, 4):
    for _prefix, _scalar in (("float", float), ("int", int), ("uint", int), ("bool", bool)):
        setattr(falcor, f"{_prefix}{_n}", _vector(f"{_prefix}{_n}", _n, _scalar))

# falcor.ui (Utils/UI/PythonUI): widgets over the testbed's DOM screen; edits queue in JS
# and are applied (with their callbacks) inside testbed.frame(), as ImGui does natively.
ui = types.ModuleType("falcor.ui")

class SliderFlags(enum.IntFlag):
    None_ = 0
    AlwaysClamp = 1 << 4
    Logarithmic = 1 << 5
    NoRoundToFormat = 1 << 6
    NoInput = 1 << 7

def _tojs(v):
    return to_js(v, dict_converter=__import__("js").Object.fromEntries)

class Widget:
    def _init(self, kind, parent, **props):
        self._screen = parent._screen
        self._props = props
        self._parent = parent
        self._children = []
        self._visible = True
        self._enabled = True
        parent._children.append(self)
        self._id = self._screen._o.create(kind, parent._id, _tojs(props))
        self._screen._widgets[self._id] = self
    def _set(self, name, value):
        self._props[name] = value
        self._screen._o.set(self._id, name, _tojs(value))
    @property
    def parent(self): return self._parent
    @parent.setter
    def parent(self, p):
        if self._parent is not None: self._parent._children.remove(self)
        self._parent = p
        if p is not None: p._children.append(self)
        self._screen._o.setParent(self._id, p._id if p is not None else None)
    @property
    def children(self): return list(self._children)
    @property
    def visible(self): return self._visible
    @visible.setter
    def visible(self, v):
        self._visible = bool(v)
        self._screen._o.set(self._id, "visible", self._visible)
    @property
    def enabled(self): return self._enabled
    @enabled.setter
    def enabled(self, v):
        self._enabled = bool(v)
        self._screen._o.set(self._id, "enabled", self._enabled)
    def _event(self, value): pass

def _prop(name, conv=None):
    return property(lambda s: s._props[name], lambda s, v: s._set(name, conv(v) if conv else v))

class Screen(Widget):
    def __init__(self, js_screen):
        self._o = js_screen
        self._screen = self
        self._id = 0
        self._parent = None
        self._children = []
        self._visible = True
        self._enabled = True
        self._widgets = {}
    def _drain(self):
        for wid, value in self._o.takeEvents().to_py():
            w = self._widgets.get(wid)
            if w is not None and w._enabled: w._event(value)

class Window(Widget):
    def __init__(self, parent, title="", position=(10.0, 10.0), size=(400.0, 400.0)):
        self._init("window", parent, title=title, position=[float(x) for x in position], size=[float(x) for x in size])
    title = _prop("title")
    position = _prop("position", lambda v: [float(x) for x in v])
    size = _prop("size", lambda v: [float(x) for x in v])
    def show(self): self.visible = True
    def close(self): self.visible = False

class Group(Widget):
    def __init__(self, parent, label=""): self._init("group", parent, label=label)
    label = _prop("label")

class Text(Widget):
    def __init__(self, parent, text=""): self._init("text", parent, text=text)
    text = _prop("text")

class ProgressBar(Widget):
    def __init__(self, parent, fraction=0.0): self._init("progress", parent, fraction=float(fraction))
    fraction = _prop("fraction", float)

class Button(Widget):
    def __init__(self, parent, label="", callback=None):
        self.callback = callback
        self._init("button", parent, label=label)
    label = _prop("label")
    def _event(self, value):
        if self.callback: self.callback()

class Property(Widget):
    label = _prop("label")
    def _event(self, value):
        self._props["value"] = self._convert(value)
        if self.change_callback: self.change_callback()
    def _convert(self, value): return value

class Checkbox(Property):
    def __init__(self, parent, label="", change_callback=None, value=False):
        self.change_callback = change_callback
        self._init("checkbox", parent, label=label, value=bool(value))
    value = _prop("value", bool)
    def _convert(self, value): return bool(value)

class Combobox(Property):
    def __init__(self, parent, label="", change_callback=None, items=(), value=0):
        self.change_callback = change_callback
        self._init("combobox", parent, label=label, items=list(items), value=int(value))
    items = _prop("items", list)
    value = _prop("value", int)
    def _convert(self, value): return int(value)

def _vector_widget(kind, n, integer, default_format):
    scalar = int if integer else float
    conv = (lambda v: scalar(v)) if n == 1 else (lambda v: [scalar(x) for x in v])
    class W(Property):
        def __init__(self, parent, label="", change_callback=None, value=None, speed=1.0, min=0, max=0, format=default_format, flags=SliderFlags.None_):
            self.change_callback = change_callback
            value = conv(value if value is not None else (0 if n == 1 else [0] * n))
            props = dict(label=label, value=value, min=scalar(min), max=scalar(max), format=format, flags=int(flags), components=n, integer=integer)
            if kind == "drag": props["speed"] = float(speed)
            self._init(kind, parent, **props)
        value = _prop("value", conv)
        min = _prop("min", scalar)
        max = _prop("max", scalar)
        format = _prop("format")
        flags = _prop("flags", int)
        def _convert(self, value): return conv(value.to_py() if hasattr(value, "to_py") else value)
    if kind == "drag": W.speed = _prop("speed", float)
    return W

for _kind, _prefix in (("drag", "Drag"), ("slider", "Slider")):
    for _n in (1, 2, 3, 4):
        for _integer, _scalar in ((False, "Float"), (True, "Int")):
            _name = f"{_prefix}{_scalar}{'' if _n == 1 else _n}"
            _cls = _vector_widget(_kind, _n, _integer, "%d" if _integer else "%.3f")
            _cls.__name__ = _name
            setattr(ui, _name, _cls)

for _cls in (Widget, Screen, Window, Group, Text, ProgressBar, Button, Property, Checkbox, Combobox, SliderFlags):
    setattr(ui, _cls.__name__, _cls)
falcor.ui = ui
sys.modules["falcor.ui"] = ui

# Pyodide has no threads: threading.Timer fires from testbed.frame(), which ui_demo.py
# notes it relies on ("run frame-by-frame to have Python's timer working").
import threading as _threading, time as _time
_timers = []
class _FrameTimer:
    def __init__(self, interval, function, args=None, kwargs=None):
        self.interval, self.function, self.args, self.kwargs = interval, function, args or [], kwargs or {}
        self._cancelled = False
    def start(self):
        self._due = _time.monotonic() + self.interval
        _timers.append(self)
    def cancel(self): self._cancelled = True
    def is_alive(self): return self in _timers
def _fire_timers():
    now = _time.monotonic()
    for t in [t for t in _timers if t._due <= now]:
        _timers.remove(t)
        if not t._cancelled: t.function(*t.args, **t.kwargs)
_threading.Timer = _FrameTimer
`;

export interface TestbedScriptResult {
    testbeds: Testbed[];
    stdout: string[];
}

/**
 * Runs the Testbed script at `scriptUrl` (served path, e.g. /Falcor/scripts/python/balls/balls.py).
 * `extraFiles` names files next to the script to copy into Pyodide's FS; string literals
 * ending in .slang/.py/.json are found automatically.
 */
export async function runTestbedScript(device: Device, scriptUrl: string, options: TestbedOptions & { extraFiles?: string[] } = {}): Promise<TestbedScriptResult> {
    const py = getPyodide() as Pyodide;
    const source = await (await fetch(scriptUrl)).text();
    const dirUrl = scriptUrl.slice(0, scriptUrl.lastIndexOf("/"));
    const root = "/testbed";
    const fsDir = `${root}${dirUrl}`;
    py.FS.mkdirTree(fsDir);
    const scriptPath = `${fsDir}/${scriptUrl.slice(scriptUrl.lastIndexOf("/") + 1)}`;
    py.FS.writeFile(scriptPath, source);
    const names = new Set(options.extraFiles ?? []);
    for (const m of source.matchAll(/["']([\w./-]+\.(?:slang|slangh|py|json))["']/g)) names.add(m[1]!);
    for (const name of names) {
        const res = await fetch(`${dirUrl}/${name}`);
        if (!res.ok) continue;
        const path = `${fsDir}/${name}`;
        py.FS.mkdirTree(path.slice(0, path.lastIndexOf("/")));
        py.FS.writeFile(path, new Uint8Array(await res.arrayBuffer()));
    }
    if (/^\s*import numpy|^\s*from numpy|^\s*import numpy as/m.test(source)) await py.loadPackage("numpy");

    const testbeds: Testbed[] = [];
    const enumMembers = (e: Record<string, string | number>) => Object.fromEntries(Object.entries(e).filter(([k, v]) => typeof v === "number" && isNaN(Number(k))));
    py.registerJsModule("_falcor_testbed_js", makeJsModule(device, options, (p) => py.FS.readFile(p, { encoding: "utf8" }), testbeds));
    const stdout: string[] = [];
    const prelude = [
        `_BIND_FLAGS = ${JSON.stringify(enumMembers(ResourceBindFlags as never))}`,
        `_FORMATS = ${JSON.stringify(enumMembers(ResourceFormat as never))}`,
        `_MEMORY_TYPES = ${JSON.stringify(enumMembers(MemoryType as never))}`,
        "_js_device = None",
    ].join("\n");
    await py.runPythonAsync(`${prelude}\n${kFalcorPython}`);
    await py.runPythonAsync(
        `import sys, runpy\nsys.path.insert(0, ${JSON.stringify(fsDir)})\ntry:\n    runpy.run_path(${JSON.stringify(scriptPath)}, run_name="__main__")\nexcept SystemExit:\n    pass\nfinally:\n    sys.path.remove(${JSON.stringify(fsDir)})\n    sys.modules.pop("falcor", None)\n    sys.modules.pop("falcor.ui", None)\n`,
    );
    return { testbeds, stdout };
}
