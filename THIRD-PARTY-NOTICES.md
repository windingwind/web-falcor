# Third-party notices

web-falcor is a WebGPU reimplementation of **NVIDIA Falcor**. It is derived from
Falcor's source and shader code, and it mirrors Falcor's module layout, APIs, and
`.slang` shaders. That upstream code is licensed by NVIDIA CORPORATION under the
BSD-3-Clause license reproduced below. Its copyright notice and license terms are
retained here as the license requires (see also the `WebFalcor/Overrides` shader
headers, which mark each file derived from an upstream Falcor shader).

web-falcor's own code is licensed separately under the MIT license — see
[LICENSE](LICENSE).

---

## NVIDIA Falcor

- Project: https://github.com/NVIDIAGameWorks/Falcor
- License: BSD-3-Clause

```
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions
are met:
  * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the distribution.
  * Neither the name of NVIDIA CORPORATION nor the names of its contributors may be used to endorse or promote products derived
    from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

Falcor in turn links several NVIDIA components under their own licenses (DLSS,
RTXGI, RTXDI, NRD). web-falcor does not vendor these; where it reimplements a
feature (e.g. RTXDI), consult the upstream license before redistributing:

- DLSS: https://github.com/NVIDIA/DLSS/blob/main/LICENSE.txt
- RTXGI: https://github.com/NVIDIAGameWorks/RTXGI/blob/main/License.txt
- RTXDI: https://github.com/NVIDIAGameWorks/RTXDI/blob/main/LICENSE.txt
- NRD: https://github.com/NVIDIAGameWorks/RayTracingDenoiser/blob/master/LICENSE.txt

`npm run setup:web` downloads the SDK shader headers Falcor's shaders include
into the untracked `Falcor/external/` tree of your checkout. They are fetched
from the vendors' public repositories at the versions Falcor pins and are not
committed to or redistributed by this repository:

- NanoVDB `PNanoVDB.h` from [OpenVDB v9.1.0](https://github.com/AcademySoftwareFoundation/openvdb/tree/v9.1.0) — MPL-2.0
- [RTXDI SDK v1.3.0](https://github.com/NVIDIAGameWorks/RTXDI/tree/v1.3.0) `rtxdi-sdk/include/rtxdi/` headers — NVIDIA RTX SDKs license (link above)
