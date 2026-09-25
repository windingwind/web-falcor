# Python ProgramDesc / ComputePass / Program API, run unchanged by native python
# (tests/oracle/out-native/program-api.json) and the web Testbed.
import falcor, json, sys

device = falcor.Device(type=falcor.DeviceType.Vulkan, gpu=0, enable_debug_layer=False) if hasattr(falcor.DeviceType, "Vulkan") else falcor.Device()
flags = falcor.ResourceBindFlags.ShaderResource | falcor.ResourceBindFlags.UnorderedAccess
buf = device.create_structured_buffer(struct_size=4, element_count=16, bind_flags=flags)
res = {}

def run(p):
    p.globals.result = buf
    p.execute(16)
    return buf.to_numpy().tolist()

# Keyword arguments: a code string and defines; a define edit recompiles.
code = """
#include "Utils/Math/MathConstants.slangh"
RWStructuredBuffer<uint> result;
[numthreads(16, 1, 1)]
void main(uint3 t: SV_DispatchThreadID) { result[t.x] = t.x * VALUE + uint(M_PI); }
"""
p = falcor.ComputePass(device, string=code, cs_entry="main", defines={"VALUE": 3})
res["string"] = run(p)
res["defines"] = p.program.defines
p.program.add_define("VALUE", "5")
res["redefined"] = run(p)
p.program.defines = {"VALUE": "7"}
res["setDefines"] = run(p)

# ProgramDesc: a named string module imported by another module.
desc = falcor.ProgramDesc()
desc.add_shader_module("Helper").add_string("public uint helper(uint x) { return x + 7; }")
desc.add_shader_module().add_string("import Helper;\nRWStructuredBuffer<uint> result;\n[numthreads(16, 1, 1)]\nvoid main(uint3 t: SV_DispatchThreadID) { result[t.x] = helper(t.x); }")
desc.cs_entry("main")
res["desc"] = run(falcor.ComputePass(device, desc))

# Type conformances: dynamic dispatch by conformance ID.
code3 = """
interface IFoo { uint get(uint x); }
struct A : IFoo { uint get(uint x) { return 100 + x; } }
struct B : IFoo { uint get(uint x) { return 200 + x; } }
RWStructuredBuffer<uint> result;
[numthreads(16, 1, 1)]
void main(uint3 t: SV_DispatchThreadID)
{
    IFoo f = createDynamicObject<IFoo, uint>(t.x % 2, 0);
    result[t.x] = f.get(t.x);
}
"""
p3 = falcor.ComputePass(device, string=code3, cs_entry="main", type_conformances={("A", "IFoo"): 0, ("B", "IFoo"): 1})
res["conformances"] = run(p3)
res["conformanceList"] = sorted([list(k) + [v] for k, v in p3.program.type_conformances.items()])
# Adding a listed pair keeps its ID; replacing the list swaps the dispatch.
p3.program.add_type_conformance("A", "IFoo", 1)
res["addExisting"] = run(p3)
p3.program.type_conformances = {("A", "IFoo"): 1, ("B", "IFoo"): 0}
res["swapped"] = run(p3)

prog = device.create_program(string=code, cs_entry="main", defines={"VALUE": True})
res["programDefines"] = prog.defines
try:
    falcor.ComputePass(device, bogus=1)
    res["unknownKwarg"] = "accepted"
except Exception as e:
    res["unknownKwarg"] = str(e).split("\n")[0]

out = sys.argv[1] if len(sys.argv) > 1 else None
if out:
    json.dump(res, open(out, "w"))
print("PROGRAMAPI " + json.dumps(res))
