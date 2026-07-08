/***************************************************************************
 # WebFalcor override of Scene/Material/BasicMaterialData.slang (docs §4.3).
 #
 # Diff vs upstream: float16_t fields are stored as scalar uint bit containers
 # with `property` accessors unpacking via f16tof32/f32tof16 (core WGSL
 # pack/unpack2x16float). All fields are 4-byte scalars so WGSL alignment rules
 # reproduce upstream's tightly packed 96-byte layout exactly — the f16->f32
 # type demotion used on devices without shader-f16 would otherwise change the
 # struct size and break MaterialPayload reinterpretation (and float3/uint2
 # members would change offsets under WGSL alignment).
 **************************************************************************/

#include "Utils/HostDeviceShared.slangh"

#ifdef HOST_CODE
#include "Scene/Material/TextureHandle.slang"
#include "Scene/Material/MaterialData.slang"
#endif

__exported import Scene.Material.TextureHandle;
__exported import Scene.Material.MaterialData;

BEGIN_NAMESPACE_FALCOR

/** Basic material data (payload byte layout identical to upstream):
      0 flags | 4 emissiveFactor | 8..15 baseColor f16x4 | 16..23 specular f16x4
     24..35 emissive float3 | 36 specularTransmission f16 + 38..43 transmission f16x3
     44 diffuseTransmission f16 + 46..51 volumeScattering f16x3
     52 _pad f16 + 54..59 volumeAbsorption f16x3 | 60 volumeAnisotropy f16 + 62 pad
     64 displacementScale | 68 displacementOffset | 72..95 six TextureHandles
*/
struct BasicMaterialData
{
    uint  flags = 0;
    float emissiveFactor = 1.f;

    uint baseColorBits0 = 0x3c003c00;  ///< baseColor.xy (f16 pair), default (1,1)
    uint baseColorBits1 = 0x3c003c00;  ///< baseColor.zw
    uint specularBits0 = 0;            ///< specular.xy
    uint specularBits1 = 0;            ///< specular.zw

    float emissiveR = 0.f;
    float emissiveG = 0.f;
    float emissiveB = 0.f;
    uint transBits0 = 0x3c000000;      ///< specularTransmission | transmission.x (default 0, 1)
    uint transBits1 = 0x3c003c00;      ///< transmission.yz (default 1,1)
    uint volBits0 = 0;                 ///< diffuseTransmission | volumeScattering.x
    uint volBits1 = 0;                 ///< volumeScattering.yz
    uint volBits2 = 0;                 ///< _pad | volumeAbsorption.x
    uint volBits3 = 0;                 ///< volumeAbsorption.yz
    uint volBits4 = 0;                 ///< volumeAnisotropy | trailing pad

    float displacementScale = 0.f;
    float displacementOffset = 0.f;

    TextureHandle texBaseColor;
    TextureHandle texSpecular;
    TextureHandle texEmissive;
    TextureHandle texNormalMap;
    TextureHandle texTransmission;
    TextureHandle texDisplacementMap;

    // ---- accessors mirroring the upstream fields ----

    property float16_t4 baseColor
    {
        get {
            return float16_t4(f16tof32(baseColorBits0 & 0xffff), f16tof32(baseColorBits0 >> 16),
                              f16tof32(baseColorBits1 & 0xffff), f16tof32(baseColorBits1 >> 16));
        }
        set {
            baseColorBits0 = f32tof16(float(newValue.x)) | (f32tof16(float(newValue.y)) << 16);
            baseColorBits1 = f32tof16(float(newValue.z)) | (f32tof16(float(newValue.w)) << 16);
        }
    }

    property float16_t4 specular
    {
        get {
            return float16_t4(f16tof32(specularBits0 & 0xffff), f16tof32(specularBits0 >> 16),
                              f16tof32(specularBits1 & 0xffff), f16tof32(specularBits1 >> 16));
        }
        set {
            specularBits0 = f32tof16(float(newValue.x)) | (f32tof16(float(newValue.y)) << 16);
            specularBits1 = f32tof16(float(newValue.z)) | (f32tof16(float(newValue.w)) << 16);
        }
    }

    property float3 emissive
    {
        get { return float3(emissiveR, emissiveG, emissiveB); }
        set { emissiveR = newValue.x; emissiveG = newValue.y; emissiveB = newValue.z; }
    }

    property float16_t specularTransmission
    {
        get { return float16_t(f16tof32(transBits0 & 0xffff)); }
        set { transBits0 = (transBits0 & 0xffff0000) | f32tof16(float(newValue)); }
    }

    property float16_t3 transmission
    {
        get { return float16_t3(f16tof32(transBits0 >> 16), f16tof32(transBits1 & 0xffff), f16tof32(transBits1 >> 16)); }
        set {
            transBits0 = (transBits0 & 0xffff) | (f32tof16(float(newValue.x)) << 16);
            transBits1 = f32tof16(float(newValue.y)) | (f32tof16(float(newValue.z)) << 16);
        }
    }

    property float16_t diffuseTransmission
    {
        get { return float16_t(f16tof32(volBits0 & 0xffff)); }
        set { volBits0 = (volBits0 & 0xffff0000) | f32tof16(float(newValue)); }
    }

    property float16_t3 volumeScattering
    {
        get { return float16_t3(f16tof32(volBits0 >> 16), f16tof32(volBits1 & 0xffff), f16tof32(volBits1 >> 16)); }
        set {
            volBits0 = (volBits0 & 0xffff) | (f32tof16(float(newValue.x)) << 16);
            volBits1 = f32tof16(float(newValue.y)) | (f32tof16(float(newValue.z)) << 16);
        }
    }

    property float16_t3 volumeAbsorption
    {
        get { return float16_t3(f16tof32(volBits2 >> 16), f16tof32(volBits3 & 0xffff), f16tof32(volBits3 >> 16)); }
        set {
            volBits2 = (volBits2 & 0xffff) | (f32tof16(float(newValue.x)) << 16);
            volBits3 = f32tof16(float(newValue.y)) | (f32tof16(float(newValue.z)) << 16);
        }
    }

    property float16_t volumeAnisotropy
    {
        get { return float16_t(f16tof32(volBits4 & 0xffff)); }
        set { volBits4 = (volBits4 & 0xffff0000) | f32tof16(float(newValue)); }
    }

    static constexpr uint kShadingModelBits = 1;
    static constexpr uint kNormalMapTypeBits = 2;
    static constexpr uint kEntryPointVolumePropertiesBits = 1;

    static constexpr uint kShadingModelOffset = 0;
    static constexpr uint kNormalMapTypeOffset = kShadingModelOffset + kShadingModelBits;
    static constexpr uint kEntryPointVolumePropertiesOffset = kNormalMapTypeOffset + kNormalMapTypeBits;
    static constexpr uint kMinSamplerIDOffset = kEntryPointVolumePropertiesOffset + kEntryPointVolumePropertiesBits;
    static constexpr uint kMaxSamplerIDOffset = kMinSamplerIDOffset + MaterialHeader::kSamplerIDBits;

    static constexpr uint kTotalFlagsBits = kMaxSamplerIDOffset + MaterialHeader::kSamplerIDBits;

    SETTER_DECL void setShadingModel(ShadingModel model) { flags = PACK_BITS(kShadingModelBits, kShadingModelOffset, flags, (uint)model); }
    ShadingModel getShadingModel() CONST_FUNCTION { return ShadingModel(EXTRACT_BITS(kShadingModelBits, kShadingModelOffset, flags)); }

    SETTER_DECL void setNormalMapType(NormalMapType type) { flags = PACK_BITS(kNormalMapTypeBits, kNormalMapTypeOffset, flags, (uint)type); }
    NormalMapType getNormalMapType() CONST_FUNCTION { return NormalMapType(EXTRACT_BITS(kNormalMapTypeBits, kNormalMapTypeOffset, flags)); }

    SETTER_DECL void setHasEntryPointVolumeProperties(bool poe) { flags = PACK_BITS(kEntryPointVolumePropertiesBits, kEntryPointVolumePropertiesOffset, flags, (uint)poe); }
    bool getHasEntryPointVolumeProperties() CONST_FUNCTION { return bool(EXTRACT_BITS(kEntryPointVolumePropertiesBits, kEntryPointVolumePropertiesOffset, flags)); }

    SETTER_DECL void setDisplacementMinSamplerID(uint samplerID) { flags = PACK_BITS(MaterialHeader::kSamplerIDBits, kMinSamplerIDOffset, flags, samplerID); }
    SETTER_DECL void setDisplacementMaxSamplerID(uint samplerID) { flags = PACK_BITS(MaterialHeader::kSamplerIDBits, kMaxSamplerIDOffset, flags, samplerID); }
    uint getDisplacementMinSamplerID() CONST_FUNCTION { return EXTRACT_BITS(MaterialHeader::kSamplerIDBits, kMinSamplerIDOffset, flags); }
    uint getDisplacementMaxSamplerID() CONST_FUNCTION { return EXTRACT_BITS(MaterialHeader::kSamplerIDBits, kMaxSamplerIDOffset, flags); }
};

END_NAMESPACE_FALCOR
