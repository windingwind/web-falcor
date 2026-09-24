// OpenSubdiv Bfr tessellation for web-falcor, following Falcor's
// Source/Modules/USDUtils/Tessellator/Tessellation.cpp (the refinement branch)
// with plain arrays in place of USD types. Built by scripts/build-opensubdiv-wasm.mjs.
#include <opensubdiv/bfr/refinerSurfaceFactory.h>
#include <opensubdiv/bfr/surface.h>
#include <opensubdiv/bfr/tessellation.h>
#include <opensubdiv/far/topologyDescriptor.h>

#include <cmath>
#include <cstring>
#include <map>
#include <vector>

using namespace OpenSubdiv;

namespace
{
enum UvInterp { kNone = 0, kVertex = 1, kVarying = 2, kFaceVarying = 3, kUniform = 4 };

// UsdIndexedVector: unique values (bitwise) in first-appearance order, plus an index per append.
template<int N>
struct IndexedSet
{
    std::map<std::vector<unsigned char>, int> lookup;
    std::vector<float> values;
    std::vector<int> indices;
    bool append(const float* v, int& idx)
    {
        std::vector<unsigned char> key((const unsigned char*)v, (const unsigned char*)v + N * sizeof(float));
        auto it = lookup.find(key);
        if (it != lookup.end())
        {
            idx = it->second;
            indices.push_back(idx);
            return false;
        }
        idx = (int)(values.size() / N);
        lookup.emplace(std::move(key), idx);
        values.insert(values.end(), v, v + N);
        indices.push_back(idx);
        return true;
    }
};

struct Result
{
    std::vector<float> positions, normals, uvs;
    std::vector<int> indices, coarseFaces;
    int uvInterp = kNone;
} gResult;

inline void cross(const float* a, const float* b, float* out)
{
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}
} // namespace

extern "C"
{
// scheme: 0 catmullClark, 1 loop, 2 bilinear. Returns the triangle count, or -1 if the refiner fails.
int osd_tessellate(
    int scheme, int level, int leftHanded, int vtxBoundary, int fvarLinear,
    const float* points, int numPoints, const int* faceCounts, int numFaces, const int* faceIndices, int numIndices,
    const float* uvs, int uvInterp
)
{
    (void)numPoints;
    gResult = Result();
    const int tessellationRate = level + 1;

    Sdc::Options options;
    options.SetVtxBoundaryInterpolation((Sdc::Options::VtxBoundaryInterpolation)vtxBoundary);
    options.SetFVarLinearInterpolation((Sdc::Options::FVarLinearInterpolation)fvarLinear);
    const Sdc::SchemeType schemes[] = {Sdc::SCHEME_CATMARK, Sdc::SCHEME_LOOP, Sdc::SCHEME_BILINEAR};
    Far::TopologyRefinerFactory<Far::TopologyDescriptor>::Options refinerOptions(schemes[scheme], options);

    Far::TopologyDescriptor::FVarChannel channels;
    Far::TopologyDescriptor desc = {};
    desc.isLeftHanded = leftHanded != 0;
    desc.numVertices = numIndices; // as natively: faceIndices.size()
    desc.numFaces = numFaces;
    desc.numVertsPerFace = faceCounts;
    desc.vertIndicesPerFace = faceIndices;
    desc.fvarChannels = &channels;

    const float* uvData = uvInterp != kNone ? uvs : nullptr;
    IndexedSet<2> indexedUVSet;
    if (uvData && uvInterp == kFaceVarying)
    {
        // OpenSubdiv expects indexed face-varying UVs.
        int idx;
        for (int i = 0; i < numIndices; ++i)
            indexedUVSet.append(uvs + 2 * i, idx);
        uvData = indexedUVSet.values.data();
        channels.numValues = (int)(indexedUVSet.values.size() / 2);
        channels.valueIndices = indexedUVSet.indices.data();
        ++desc.numFVarChannels;
    }

    std::unique_ptr<Far::TopologyRefiner> refiner(Far::TopologyRefinerFactory<Far::TopologyDescriptor>::Create(desc, refinerOptions));
    if (!refiner)
        return -1;

    Bfr::RefinerSurfaceFactory<>::Options surfaceOptions;
    Bfr::RefinerSurfaceFactory<> surfaceFactory(*refiner, surfaceOptions);
    Bfr::Surface<float> vertexSurface, varyingSurface, fvarSurface;

    std::vector<float> facePatchPoints, outCoords, outPos, outNormals, outUV;
    std::vector<int> outFacets;
    Bfr::Tessellation::Options tessOptions;
    tessOptions.SetFacetSize(3);

    // MeshIndexer: positions shared bitwise; normals follow positions.
    IndexedSet<3> positionSet;
    gResult.uvInterp = uvInterp;
    bool createVaryingSurf = false;
    Bfr::Surface<float>* uvSurface = nullptr;
    if (uvInterp == kFaceVarying)
        uvSurface = &fvarSurface;
    else if (uvInterp == kVertex)
        uvSurface = &vertexSurface;
    else if (uvInterp == kVarying)
    {
        uvSurface = &varyingSurface;
        createVaryingSurf = true;
    }
    Bfr::SurfaceFactoryMeshAdapter::FVarID fvarID = 0;

    const int faceCount = surfaceFactory.GetNumFaces();
    for (int f = 0; f < faceCount; ++f)
    {
        surfaceFactory.InitSurfaces(f, &vertexSurface, &fvarSurface, &fvarID, desc.numFVarChannels, createVaryingSurf ? &varyingSurface : nullptr);
        if (!vertexSurface.IsValid())
            continue;
        if (uvSurface && !uvSurface->IsValid())
            uvSurface = &vertexSurface;

        Bfr::Tessellation tessPattern(vertexSurface.GetParameterization(), tessellationRate, tessOptions);
        const int outCoordCount = tessPattern.GetNumCoords();
        outCoords.resize(outCoordCount * 2);
        tessPattern.GetCoords(outCoords.data());
        outNormals.resize(outCoordCount * 3);
        outPos.resize(outCoordCount * 3);

        if (uvSurface)
        {
            facePatchPoints.resize(uvSurface->GetNumPatchPoints() * 2);
            outUV.resize(outCoordCount * 2);
            uvSurface->PreparePatchPoints(uvData, 2, facePatchPoints.data(), 2);
            for (int i = 0; i < outCoordCount; ++i)
                uvSurface->Evaluate(&outCoords[i * 2], facePatchPoints.data(), 2, &outUV[i * 2]);
        }
        else if (uvInterp == kUniform)
        {
            outUV.assign(uvs + 2 * f, uvs + 2 * f + 2);
        }

        facePatchPoints.resize(vertexSurface.GetNumPatchPoints() * 3);
        vertexSurface.PreparePatchPoints(points, 3, facePatchPoints.data(), 3);
        for (int i = 0; i < outCoordCount; ++i)
        {
            float du[3], dv[3], n[3];
            vertexSurface.Evaluate(&outCoords[i * 2], facePatchPoints.data(), 3, &outPos[i * 3], du, dv);
            cross(du, dv, n);
            const float len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
            for (int k = 0; k < 3; ++k)
                outNormals[i * 3 + k] = n[k] / len;
        }

        const int facetCount = tessPattern.GetNumFacets();
        outFacets.resize(facetCount * 3);
        tessPattern.GetFacets(outFacets.data());

        // MeshIndexer::addFacets
        std::vector<int> indexMap;
        for (int i = 0; i < outCoordCount; ++i)
        {
            int idx;
            if (positionSet.append(&outPos[i * 3], idx))
            {
                gResult.normals.insert(gResult.normals.end(), &outNormals[i * 3], &outNormals[i * 3] + 3);
                if (uvInterp == kVertex || uvInterp == kVarying)
                    gResult.uvs.insert(gResult.uvs.end(), &outUV[i * 2], &outUV[i * 2] + 2);
            }
            indexMap.push_back(idx);
        }
        for (int idx : outFacets)
        {
            gResult.indices.push_back(indexMap[idx]);
            if (uvInterp == kFaceVarying)
                gResult.uvs.insert(gResult.uvs.end(), &outUV[idx * 2], &outUV[idx * 2] + 2);
        }
        if (uvInterp == kUniform)
            gResult.uvs.insert(gResult.uvs.end(), outUV.begin(), outUV.begin() + 2);
        for (int i = 0; i < facetCount; ++i)
            gResult.coarseFaces.push_back(f);
    }
    gResult.positions = positionSet.values;
    return (int)(gResult.indices.size() / 3);
}

const float* osd_positions() { return gResult.positions.data(); }
int osd_position_count() { return (int)(gResult.positions.size() / 3); }
const float* osd_normals() { return gResult.normals.data(); }
const float* osd_uvs() { return gResult.uvs.data(); }
int osd_uv_count() { return (int)(gResult.uvs.size() / 2); }
const int* osd_indices() { return gResult.indices.data(); }
const int* osd_coarse_faces() { return gResult.coarseFaces.data(); }
}
