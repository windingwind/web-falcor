// MikkTSpace (Falcor/external/mikktspace) behind one call, for packages/falcor/wasm/mikktspace.wasm.
// Mirrors SceneBuilder's MikkTSpaceWrapper: face-varying inputs (3 corners per face), and each
// tangent normalized as v * (1 / sqrt(dot(v, v))) with the MikkTSpace sign in w.
#include <math.h>
#include <stdlib.h>
#include "mikktspace.h"

typedef struct {
    int faceCount;
    const float* positions; // float3 per corner
    const float* normals;   // float3 per corner
    const float* texCrds;   // float2 per corner
    float* tangents;        // float4 per corner (output)
} Mesh;

static Mesh* mesh(const SMikkTSpaceContext* c) { return (Mesh*)c->m_pUserData; }
static int getNumFaces(const SMikkTSpaceContext* c) { return mesh(c)->faceCount; }
static int getNumVerticesOfFace(const SMikkTSpaceContext* c, const int face) { return 3; }
static void getPosition(const SMikkTSpaceContext* c, float out[], const int face, const int vert)
{
    const float* p = mesh(c)->positions + (face * 3 + vert) * 3;
    out[0] = p[0], out[1] = p[1], out[2] = p[2];
}
static void getNormal(const SMikkTSpaceContext* c, float out[], const int face, const int vert)
{
    const float* n = mesh(c)->normals + (face * 3 + vert) * 3;
    out[0] = n[0], out[1] = n[1], out[2] = n[2];
}
static void getTexCoord(const SMikkTSpaceContext* c, float out[], const int face, const int vert)
{
    const float* t = mesh(c)->texCrds + (face * 3 + vert) * 2;
    out[0] = t[0], out[1] = t[1];
}
static void setTSpaceBasic(const SMikkTSpaceContext* c, const float t[], const float sign, const int face, const int vert)
{
    float* out = mesh(c)->tangents + (face * 3 + vert) * 4;
    const float r = 1.f / sqrtf(t[0] * t[0] + t[1] * t[1] + t[2] * t[2]);
    out[0] = t[0] * r, out[1] = t[1] * r, out[2] = t[2] * r, out[3] = sign;
}

// Returns 1 on success (genTangSpaceDefault).
int mikk_generate(int faceCount, const float* positions, const float* normals, const float* texCrds, float* tangents)
{
    Mesh m = {faceCount, positions, normals, texCrds, tangents};
    SMikkTSpaceInterface iface = {0};
    iface.m_getNumFaces = getNumFaces;
    iface.m_getNumVerticesOfFace = getNumVerticesOfFace;
    iface.m_getPosition = getPosition;
    iface.m_getNormal = getNormal;
    iface.m_getTexCoord = getTexCoord;
    iface.m_setTSpaceBasic = setTSpaceBasic;
    SMikkTSpaceContext ctx = {&iface, &m};
    return genTangSpaceDefault(&ctx) ? 1 : 0;
}

void* mikk_alloc(int bytes) { return malloc(bytes); }
void mikk_free(void* p) { free(p); }
