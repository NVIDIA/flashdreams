/*
 *  Copyright (c) 2009-2011, NVIDIA Corporation
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *      * Neither the name of NVIDIA Corporation nor the
 *        names of its contributors may be used to endorse or promote products
 *        derived from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once
#include "3d/Texture.hpp"
#include "gpu/GLContext.hpp"

namespace FW
{
//------------------------------------------------------------------------

class MeshBase
{
public:
    enum AttribType // allows arbitrary values
    {
        AttribType_Position = 0,    // (x, y, z) or (x, y, z, w)
        AttribType_Normal,          // (x, y, z)
        AttribType_Color,           // (r, g, b) or (r, g, b, a)
        AttribType_TexCoord,        // (u, v) or (u, v, w)

        AttribType_AORadius,        // (min, max)

        AttribType_Max
    };

    enum AttribFormat
    {
        AttribFormat_U8 = 0,
        AttribFormat_S32,
        AttribFormat_F32,

        AttribFormat_Max
    };

    enum TextureType
    {
        TextureType_Diffuse = 0,    // Diffuse color map.
        TextureType_Alpha,          // Alpha map (green = opacity).
        TextureType_Displacement,   // Displacement map (green = height).
        TextureType_Normal,         // Tangent-space normal map.
        TextureType_Environment,    // Environment map (spherical coordinates).

        TextureType_Max
    };

    struct AttribSpec
    {
        AttribType      type;
        AttribFormat    format;
        S32             length;
        S32             offset;
        S32             bytes;
    };

    struct Material
    {
        Vec4f           diffuse;
        Vec3f           specular;
        F32             glossiness;
        F32             displacementCoef; // height = texture/255 * coef + bias
        F32             displacementBias;
        Texture         textures[TextureType_Max];

        Material(void)
        {
            diffuse             = Vec4f(0.75f, 0.75f, 0.75f, 1.0f);
            specular            = 0.5f;
            glossiness          = 32.0f;
            displacementCoef    = 1.0f;
            displacementBias    = 0.0f;
        }
    };

private:
    struct Submesh
    {
        Array<Vec3i>*   indices;
        Material        material;
        S32             ofsInVBO;
        S32             sizeInVBO;
    };

public:
                        MeshBase            (void)                          { init(); }
                        MeshBase            (const MeshBase& other)         { init(); addAttribs(other); set(other); }
                        ~MeshBase           (void)                          { clear(); }

    int                 addAttrib           (AttribType type, AttribFormat format, int length);
    void                addAttribs          (const MeshBase& other);
    int                 numAttribs          (void) const                    { return m_attribs.getSize(); }
    const AttribSpec&   attribSpec          (int attrib) const              { return m_attribs[attrib]; }
    int                 findAttrib          (AttribType type) const         { return findNextAttrib(type, -1); }
    int                 findNextAttrib      (AttribType type, int prevAttrib) const;
    bool                isCompatible        (const MeshBase& other) const;

    void                clear               (void)                          { m_isInMemory = true; clearVertices(); clearSubmeshes(); }
    void                set                 (const MeshBase& other);
    void                append              (const MeshBase& other);
    void                compact             (void);

    int                 numVertices         (void) const                    { return m_numVertices; }
    int                 vertexStride        (void) const                    { return m_stride; }
    void                resetVertices       (int num);
    void                clearVertices       (void)                          { resizeVertices(0); }
    void                resizeVertices      (int num);
    const U8*           getVertexPtr        (int idx = 0) const             { FW_ASSERT(isInMemory() && idx >= 0 && idx <= numVertices()); return m_vertices.getPtr() + idx * m_stride; }
    U8*                 getMutableVertexPtr (int idx = 0)                   { FW_ASSERT(isInMemory() && idx >= 0 && idx <= numVertices()); freeVBO(); return m_vertices.getPtr() + idx * m_stride; }
    const U8*           vertex              (int idx) const                 { FW_ASSERT(isInMemory() && idx >= 0 && idx < numVertices()); return getVertexPtr(idx); }
    U8*                 mutableVertex       (int idx)                       { FW_ASSERT(isInMemory() && idx >= 0 && idx < numVertices()); return getMutableVertexPtr(idx); }
    void                setVertex           (int idx, const void* ptr)      { setVertices(idx, ptr, 1); }
    void                setVertices         (int idx, const void* ptr, int num) { FW_ASSERT(ptr && num >= 0 && idx + num <= numVertices()); memcpy(getMutableVertexPtr(idx), ptr, num * m_stride); }
    U8*                 addVertex           (const void* ptr = NULL)        { return addVertices(ptr, 1); }
    U8*                 addVertices         (const void* ptr, int num)      { FW_ASSERT(isInMemory() && num >= 0); freeVBO(); m_numVertices += num; U8* slot = m_vertices.add(NULL, num * m_stride); if (ptr) memcpy(slot, ptr, num * m_stride); return slot; }
    Vec4f               getVertexAttrib     (int idx, int attrib) const;
    void                setVertexAttrib     (int idx, int attrib, const Vec4f& v);

    int                 numSubmeshes        (void) const                    { return m_submeshes.getSize(); }
    int                 numTriangles        (void) const                    { int res = 0; for (int i = 0; i < m_submeshes.getSize(); i++) res += m_submeshes[i].indices->getSize(); return res; }
    void                resizeSubmeshes     (int num);
    void                clearSubmeshes      (void)                          { resizeSubmeshes(0); }
    const Array<Vec3i>& indices             (int submesh) const             { FW_ASSERT(isInMemory()); return *m_submeshes[submesh].indices; }
    Array<Vec3i>&       mutableIndices      (int submesh)                   { FW_ASSERT(isInMemory()); freeVBO(); return *m_submeshes[submesh].indices; }
    void                setIndices          (int submesh, const Vec3i* ptr, int size) { mutableIndices(submesh).set(ptr, size); }
    void                setIndices          (int submesh, const S32* ptr, int size) { FW_ASSERT(size % 3 == 0); mutableIndices(submesh).set((const Vec3i*)ptr, size / 3); }
    void                setIndices          (int submesh, const Array<Vec3i>& v) { mutableIndices(submesh).set(v); }
    const Material&     material            (int submesh) const             { return m_submeshes[submesh].material; }
    Material&           material            (int submesh)                   { return m_submeshes[submesh].material; }
    int                 addSubmesh          (void)                          { int num = numSubmeshes(); resizeSubmeshes(num + 1); return num; }

    Buffer&             getVBO              (void);
    int                 vboAttribOffset     (int attrib)                    { getVBO(); return attribSpec(attrib).offset; }
    int                 vboAttribStride     (int attrib)                    { getVBO(); FW_ASSERT(attrib >= 0 && attrib < m_attribs.getSize()); FW_UNREF(attrib); return vertexStride(); }
    int                 vboIndexOffset      (int submesh)                   { getVBO(); return m_submeshes[submesh].ofsInVBO; }
    int                 vboIndexSize        (int submesh)                   { getVBO(); return m_submeshes[submesh].sizeInVBO; }

    void                setGLAttrib         (GLContext* gl, int attrib, int loc);
    void                draw                (GLContext* gl, const Mat4f& posToCamera, const Mat4f& projection, GLContext::Program* prog = NULL, bool gouraud = false);

    bool                isInMemory          (void) const                    { return m_isInMemory; }
    void                freeMemory          (void);
    bool                isInVBO             (void) const                    { return m_isInVBO; }
    void                freeVBO             (void)                          { m_vbo.reset(); m_isInVBO = false; }

    void                xformPositions      (const Mat4f& mat);
    void                xformNormals        (const Mat3f& mat, bool normalize = true);
    void                xform               (const Mat4f& mat)              { xformPositions(mat); xformNormals(mat.getXYZ().transposed().inverted()); }

    void                getBBox             (Vec3f& lo, Vec3f& hi) const;
    void                recomputeNormals    (void);
    void                flipTriangles       (void);
    void                clean               (void);                         // Remove empty submeshes, degenerate triangles, and unreferenced vertices.
    void                collapseVertices    (void);                         // Collapse duplicate vertices.
    void                dupVertsPerSubmesh  (void);                         // If a vertex is shared between multiple submeshes, duplicate it for each.
    void                fixMaterialColors   (void);                         // If a material is textured, override diffuse color with average over texels.
    void                simplify            (F32 maxError);                 // Collapse short edges. Do not allow vertices to drift more than maxError.

    const U8*           operator[]          (int vidx) const                { return vertex(vidx); }
    U8*                 operator[]          (int vidx)                      { return mutableVertex(vidx); }
    MeshBase&           operator=           (const MeshBase& other)         { set(other); return *this; }
    MeshBase&           operator+=          (const MeshBase& other)         { append(other); return *this; }

private:
    void                init                (void)                          { m_stride = 0; m_numVertices = 0; m_isInMemory = true; m_isInVBO = false; }

private:
    S32                 m_stride;           // Bytes per vertex in m_vertices and m_vbo.
    S32                 m_numVertices;      // Total number of vertices.
    bool                m_isInMemory;       // Whether m_vertices and m_submeshes[].indices are valid.
    bool                m_isInVBO;          // Whether m_vbo is valid.

    Array<AttribSpec>   m_attribs;
    Array<U8>           m_vertices;
    Array<Submesh>      m_submeshes;
    Buffer              m_vbo;
};

//------------------------------------------------------------------------

template <class V> class Mesh : public MeshBase
{
public:
                        Mesh                (void)                          { V::listAttribs(*this); FW_ASSERT(vertexStride() == sizeof(V)); }
                        Mesh                (const MeshBase& other)         { V::listAttribs(*this); FW_ASSERT(vertexStride() == sizeof(V)); set(other); }
                        ~Mesh               (void)                          {}

    const V*            getVertexPtr        (int idx = 0) const             { return (V*)MeshBase::getVertexPtr(idx); }
    V*                  getMutableVertexPtr (int idx = 0)                   { return (V*)MeshBase::getMutableVertexPtr(idx); }
    const V&            vertex              (int idx) const                 { return *getVertexPtr(idx); }
    V&                  mutableVertex       (int idx)                       { return *getMutableVertexPtr(idx); }
    void                setVertex           (int idx, const V& value)       { setVertices(idx, &value, 1); }
    void                setVertices         (int idx, const V* ptr, int num) { FW_ASSERT(ptr && num >= 0 && idx + num <= numVertices()); V* slot = getMutableVertexPtr(idx); for (int i = 0; i < num; i++) slot[i] = ptr[i]; }
    V&                  addVertex           (const V& value)                { V* slot = (V*)MeshBase::addVertex(NULL); *slot = value; return *slot; }
    V&                  addVertex           (void)                          { return *(V*)MeshBase::addVertex(NULL); }
    V*                  addVertices         (const V* ptr, int num)         { V* slot = (V*)MeshBase::addVertices(ptr, num); if (ptr) for (int i = 0; i < num; i++) slot[i] = ptr[i]; return slot; }

    const V&            operator[]          (int vidx) const                { return vertex(vidx); }
    V&                  operator[]          (int vidx)                      { return mutableVertex(vidx); }
    Mesh&               operator=           (const Mesh& other)             { set(other); return *this; }
};

//------------------------------------------------------------------------

struct VertexP
{
    Vec3f               p;

    VertexP(void) {}
    VertexP(const Vec3f& pp) : p(pp) {}

    static void listAttribs(MeshBase& mesh)
    {
        mesh.addAttrib(MeshBase::AttribType_Position,   MeshBase::AttribFormat_F32, 3);
    }
};

//------------------------------------------------------------------------

struct VertexPN
{
    Vec3f               p;
    Vec3f               n;

    VertexPN(void) {}
    VertexPN(const Vec3f& pp, const Vec3f& nn) : p(pp), n(nn) {}

    static void listAttribs(MeshBase& mesh)
    {
        mesh.addAttrib(MeshBase::AttribType_Position,   MeshBase::AttribFormat_F32, 3);
        mesh.addAttrib(MeshBase::AttribType_Normal,     MeshBase::AttribFormat_F32, 3);
    }
};

//------------------------------------------------------------------------

struct VertexPNC
{
    Vec3f               p;
    Vec3f               n;
    Vec4f               c;

    VertexPNC(void) {}
    VertexPNC(const Vec3f& pp, const Vec3f& nn, const Vec4f& cc) : p(pp), n(nn), c(cc) {}

    static void listAttribs(MeshBase& mesh)
    {
        mesh.addAttrib(MeshBase::AttribType_Position,   MeshBase::AttribFormat_F32, 3);
        mesh.addAttrib(MeshBase::AttribType_Normal,     MeshBase::AttribFormat_F32, 3);
        mesh.addAttrib(MeshBase::AttribType_Color,      MeshBase::AttribFormat_F32, 4);
    }
};

//------------------------------------------------------------------------

struct VertexPNT
{
    Vec3f               p;
    Vec3f               n;
    Vec2f               t;

    VertexPNT(void) {}
    VertexPNT(const Vec3f& pp, const Vec3f& nn, const Vec2f& tt) : p(pp), n(nn), t(tt) {}

    static void listAttribs(MeshBase& mesh)
    {
        mesh.addAttrib(MeshBase::AttribType_Position,   MeshBase::AttribFormat_F32, 3);
        mesh.addAttrib(MeshBase::AttribType_Normal,     MeshBase::AttribFormat_F32, 3);
        mesh.addAttrib(MeshBase::AttribType_TexCoord,   MeshBase::AttribFormat_F32, 2);
    }
};

//------------------------------------------------------------------------

MeshBase*   importMesh          (const String& fileName);
void        exportMesh          (const String& fileName, const MeshBase* mesh);
String      getMeshImportFilter (void);
String      getMeshExportFilter (void);

void        addCubeToMesh   (Mesh<VertexPNC>& mesh, int submesh, const Vec3f& lo, const Vec3f& hi, const Vec4f& color, bool forceNormal = false, const Vec3f& normal = Vec3f(0.0f));

//------------------------------------------------------------------------
}
