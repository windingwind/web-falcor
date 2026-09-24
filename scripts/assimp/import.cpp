// Assimp import for web-falcor: native's Assimp version with caller-chosen post-process flags
// (AssimpImporter / TriangleMesh pass theirs) over an in-memory file set, exported as assjson.
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
    Assimp::Exporter exporter;
    const aiExportDataBlob* blob = exporter.ExportToBlob(scene, "assjson");
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
void ai_free_result() { std::string().swap(gResult); }
}
