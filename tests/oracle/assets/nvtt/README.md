NVTT reference encodings for `ImageIO.saveToDDS` (tests/gpu/suites/imageio-dds.gpu.test.ts).
`src-ldr.png` and `src-hdr.hdr` (Radiance RGBE) are the inputs (`src-hdr.dds` is the same HDR image as
RGBA32F, for round-trip tests); the `nvtt-*.dds` files come from
NVIDIA Texture Tools 3.1.6 (Falcor's packman `nvtt`):

    nvcompress -nomips -bc1|-bc2|-bc3|-bc4|-bc5|-bc7 src-ldr.png nvtt-bcN.dds
    nvcompress -nomips -bc7 src-ldr-opaque.png nvtt-bc7-opaque.dds   # src-ldr.png with alpha dropped
    nvcompress -nomips -bc6s src-hdr.hdr nvtt-bc6.dds

(NVTT reads RGBA32F DDS input as zeros, hence the .hdr source for BC6.)
