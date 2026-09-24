// Assimp import for web-falcor: native's Assimp version with caller-chosen post-process flags
// (AssimpImporter / TriangleMesh pass theirs) over an in-memory file set, exported as assjson.
// Mesh vertex/triangle arrays travel as a binary blob (ai_mesh_blob) instead: as JSON text they
// were most of BistroExterior's 450 MB export.
// Built by scripts/build-assimp-wasm.mjs.
#include <assimp/Exporter.hpp>
#include <assimp/IOStream.hpp>
#include <assimp/IOSystem.hpp>
#include <assimp/Importer.hpp>
#include <assimp/config.h>
#include <assimp/scene.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace
{
std::map<std::string, std::vector<uint8_t>> gFiles;
std::string gResult, gError;
std::vector<uint32_t> gMeshBlob;

// Lower-cased file name without directories (assets reference sidecars by relative paths).
std::string baseName(std::string path)
{
    std::replace(path.begin(), path.end(), '\\', '/');
    const size_t slash = path.find_last_of('/');
    if (slash != std::string::npos) path = path.substr(slash + 1);
    std::transform(path.begin(), path.end(), path.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    return path;
}

const std::vector<uint8_t>* findFile(const char* path)
{
    auto it = gFiles.find(baseName(path));
    return it == gFiles.end() ? nullptr : &it->second;
}

class MemoryStream : public Assimp::IOStream
{
public:
    explicit MemoryStream(const std::vector<uint8_t>& data) : mData(data) {}
    size_t Read(void* buffer, size_t size, size_t count) override
    {
        const size_t bytes = std::min(size * count, mData.size() - mPos);
        std::memcpy(buffer, mData.data() + mPos, bytes);
        mPos += bytes;
        return size ? bytes / size : 0;
    }
    size_t Write(const void*, size_t, size_t) override { return 0; }
    aiReturn Seek(size_t offset, aiOrigin origin) override
    {
        const size_t base = origin == aiOrigin_SET ? 0 : origin == aiOrigin_CUR ? mPos : mData.size();
        if (base + offset > mData.size()) return aiReturn_FAILURE;
        mPos = base + offset;
        return aiReturn_SUCCESS;
    }
    size_t Tell() const override { return mPos; }
    size_t FileSize() const override { return mData.size(); }
    void Flush() override {}

private:
    const std::vector<uint8_t>& mData;
    size_t mPos = 0;
};

class MemoryIOSystem : public Assimp::IOSystem
{
public:
    bool Exists(const char* path) const override { return findFile(path) != nullptr; }
    char getOsSeparator() const override { return '/'; }
    Assimp::IOStream* Open(const char* path, const char* mode) override
    {
        if (mode && std::strchr(mode, 'w')) return nullptr;
        const auto* data = findFile(path);
        return data ? new MemoryStream(*data) : nullptr;
    }
    void Close(Assimp::IOStream* stream) override { delete stream; }
};

// Per mesh: numVertices, hasNormals, numUVChannels, the channels' component counts,
// triangleIndexCount, then positions, normals, each channel's UVs (float32) and the
// indices of 3-index faces (uint32).
void packMeshes(const aiScene* scene)
{
    gMeshBlob.clear();
    auto pushFloats = [](const float* data, size_t count)
    {
        const size_t at = gMeshBlob.size();
        gMeshBlob.resize(at + count);
        std::memcpy(gMeshBlob.data() + at, data, count * 4);
    };
    gMeshBlob.push_back(scene->mNumMeshes);
    for (unsigned m = 0; m < scene->mNumMeshes; m++)
    {
        const aiMesh* mesh = scene->mMeshes[m];
        const unsigned n = mesh->mNumVertices;
        unsigned channels = 0;
        while (channels < AI_MAX_NUMBER_OF_TEXTURECOORDS && mesh->mTextureCoords[channels])
            channels++;
        unsigned triIndices = 0;
        for (unsigned f = 0; f < mesh->mNumFaces; f++)
            if (mesh->mFaces[f].mNumIndices == 3)
                triIndices += 3;
        gMeshBlob.push_back(n);
        gMeshBlob.push_back(mesh->mNormals ? 1 : 0);
        gMeshBlob.push_back(channels);
        // Component counts as assjson writes them (0 means 2).
        auto uvComponents = [&](unsigned c) { return mesh->mNumUVComponents[c] ? mesh->mNumUVComponents[c] : 2u; };
        for (unsigned c = 0; c < channels; c++)
            gMeshBlob.push_back(uvComponents(c));
        gMeshBlob.push_back(triIndices);
        pushFloats(&mesh->mVertices[0].x, size_t(n) * 3);
        if (mesh->mNormals)
            pushFloats(&mesh->mNormals[0].x, size_t(n) * 3);
        for (unsigned c = 0; c < channels; c++)
        {
            const unsigned comps = uvComponents(c);
            for (unsigned i = 0; i < n; i++)
                pushFloats(&mesh->mTextureCoords[c][i].x, comps);
        }
        for (unsigned f = 0; f < mesh->mNumFaces; f++)
            if (mesh->mFaces[f].mNumIndices == 3)
                for (unsigned k = 0; k < 3; k++)
                    gMeshBlob.push_back(mesh->mFaces[f].mIndices[k]);
    }
}
} // namespace

extern "C"
{
void ai_clear_files() { gFiles.clear(); }

void ai_add_file(const char* name, const uint8_t* data, int size) { gFiles[baseName(name)].assign(data, data + size); }

// Imports `mainFile` with the given aiPostProcessSteps and AI_CONFIG_PP_RVC_FLAGS; 1 on success.
int ai_import(const char* mainFile, unsigned flags, int removeComponents)
{
    gResult.clear();
    gError.clear();
    Assimp::Importer importer;
    importer.SetIOHandler(new MemoryIOSystem());
    importer.SetPropertyInteger(AI_CONFIG_PP_RVC_FLAGS, removeComponents);
    const aiScene* scene = importer.ReadFile(mainFile, flags);
    if (!scene)
    {
        gError = importer.GetErrorString();
        return 0;
    }
    packMeshes(scene);
    // Export everything else as assjson, with the mesh arrays cut to one element (restored afterwards;
    // empty meshes corrupt the exporter's scene copy).
    aiScene* mutableScene = const_cast<aiScene*>(scene);
    std::vector<std::pair<unsigned, unsigned>> counts;
    for (unsigned m = 0; m < mutableScene->mNumMeshes; m++)
    {
        aiMesh* mesh = mutableScene->mMeshes[m];
        counts.emplace_back(mesh->mNumVertices, mesh->mNumFaces);
        mesh->mNumVertices = std::min(mesh->mNumVertices, 1u);
        mesh->mNumFaces = std::min(mesh->mNumFaces, 1u);
    }
    // The exporter would first re-verbosify a non-verbose (joined) scene, which misbehaves on the
    // emptied meshes; there are no vertices left to expand.
    const unsigned sceneFlags = mutableScene->mFlags;
    mutableScene->mFlags &= ~AI_SCENE_FLAGS_NON_VERBOSE_FORMAT;
    Assimp::Exporter exporter;
    const aiExportDataBlob* blob = exporter.ExportToBlob(scene, "assjson");
    mutableScene->mFlags = sceneFlags;
    for (unsigned m = 0; m < mutableScene->mNumMeshes; m++)
    {
        mutableScene->mMeshes[m]->mNumVertices = counts[m].first;
        mutableScene->mMeshes[m]->mNumFaces = counts[m].second;
    }
    if (!blob)
    {
        gError = exporter.GetErrorString();
        return 0;
    }
    gResult.assign((const char*)blob->data, blob->size);
    return 1;
}

const char* ai_result() { return gResult.data(); }
int ai_result_size() { return (int)gResult.size(); }
const char* ai_error() { return gError.c_str(); }
const uint32_t* ai_mesh_blob() { return gMeshBlob.data(); }
int ai_mesh_blob_size() { return (int)gMeshBlob.size(); }
void ai_free_result()
{
    std::string().swap(gResult);
    std::vector<uint32_t>().swap(gMeshBlob);
}
}
