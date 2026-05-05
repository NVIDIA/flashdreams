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

#include "3d/Mesh.hpp"
#include "io/File.hpp"
#include "io/MeshBinaryIO.hpp"
#include "io/MeshWavefrontIO.hpp"
#include "base/UnionFind.hpp"
#include "base/BinaryHeap.hpp"

using namespace FW;

//------------------------------------------------------------------------

int MeshBase::addAttrib(AttribType type, AttribFormat format, int length)
{
    FW_ASSERT(format >= 0 && format < AttribFormat_Max);
    FW_ASSERT(length >= 1 && length <= 4);
    FW_ASSERT(!numVertices());

    AttribSpec spec;
    spec.type   = type;
    spec.format = format;
    spec.length = length;
    spec.offset = m_stride;

    switch (format)
    {
    case AttribFormat_U8:   spec.bytes = length * sizeof(U8); break;
    case AttribFormat_S32:  spec.bytes = length * sizeof(S32); break;
    case AttribFormat_F32:  spec.bytes = length * sizeof(F32); break;
    default:                FW_ASSERT(false); break;
    }

    m_attribs.add(spec);
    m_stride += spec.bytes;
    return m_attribs.getSize() - 1;
}

//------------------------------------------------------------------------

void MeshBase::addAttribs(const MeshBase& other)
{
    int num = other.numAttribs();
    for (int i = 0; i < num; i++)
    {
        const AttribSpec& spec = other.attribSpec(i);
        addAttrib(spec.type, spec.format, spec.length);
    }
}

//------------------------------------------------------------------------

int MeshBase::findNextAttrib(AttribType type, int prevAttrib) const
{
    for (int i = prevAttrib + 1; i < numAttribs(); i++)
        if (attribSpec(i).type == type)
            return i;
    return -1;
}

//------------------------------------------------------------------------

bool MeshBase::isCompatible(const MeshBase& other) const
{
    if (numAttribs() != other.numAttribs())
        return false;

    for (int i = 0; i < numAttribs(); i++)
    {
        const AttribSpec& a = attribSpec(i);
        const AttribSpec& b = other.attribSpec(i);
        if (a.type != b.type || a.format != b.format || a.length != b.length)
            return false;
    }
    return true;
}

//------------------------------------------------------------------------

void MeshBase::set(const MeshBase& other)
{
    if (&other == this)
        return;

    FW_ASSERT(other.isInMemory());
    clear();

    if (!isCompatible(other))
    {
        append(other);
        compact();
        return;
    }

    m_stride = other.m_stride;
    m_numVertices = other.m_numVertices;
    m_attribs = other.m_attribs;
    m_vertices = other.m_vertices;

    resizeSubmeshes(other.m_submeshes.getSize());
    for (int i = 0; i < m_submeshes.getSize(); i++)
    {
        *m_submeshes[i].indices = *other.m_submeshes[i].indices;
        m_submeshes[i].material = other.m_submeshes[i].material;
    }
}

//------------------------------------------------------------------------

void MeshBase::append(const MeshBase& other)
{
    FW_ASSERT(&other != this);
    FW_ASSERT(isInMemory());
    FW_ASSERT(other.isInMemory());

    Array<bool> dstAttribUsed(NULL, numAttribs());
    for (int i = 0; i < numAttribs(); i++)
        dstAttribUsed[i] = false;

    Array<Vec2i> copy;
    Array<Vec2i> convert;
    for (int i = 0; i < other.numAttribs(); i++)
    {
        const AttribSpec& src = other.attribSpec(i);
        for (int j = 0; j < numAttribs(); j++)
        {
            const AttribSpec& dst = attribSpec(j);
            if (src.type != dst.type || dstAttribUsed[j])
                continue;

            if (src.format != dst.format || src.length != dst.length)
                convert.add(Vec2i(i, j));
            else
            {
                for (int k = 0; k < src.bytes; k++)
                    copy.add(Vec2i(src.offset + k, dst.offset + k));
            }
            dstAttribUsed[j] = true;
            break;
        }
    }

    int oldNumVertices = m_numVertices;
    resizeVertices(oldNumVertices + other.m_numVertices);
    for (int i = 0; i < other.m_numVertices; i++)
    {
        if (copy.getSize())
        {
            const U8* src = &other.m_vertices[i * other.m_stride];
            U8* dst = &m_vertices[(i + oldNumVertices) * m_stride];
            for (int j = 0; j < copy.getSize(); j++)
                dst[copy[j].y] = src[copy[j].x];
        }
        for (int j = 0; j < convert.getSize(); j++)
            setVertexAttrib(i + oldNumVertices, convert[j].y,
                other.getVertexAttrib(i, convert[j].x));
    }

    int oldNumSubmeshes = numSubmeshes();
    resizeSubmeshes(oldNumSubmeshes + other.numSubmeshes());
    for (int i = 0; i < other.numSubmeshes(); i++)
    {
        const Submesh& src = other.m_submeshes[i];
        Submesh& dst = m_submeshes[i + oldNumSubmeshes];

        dst.indices->reset(src.indices->getSize());
        for (int j = 0; j < src.indices->getSize(); j++)
            dst.indices->set(j, src.indices->get(j) + oldNumVertices);
        dst.material = src.material;
    }
}

//------------------------------------------------------------------------

void MeshBase::compact(void)
{
    m_attribs.compact();
    m_vertices.compact();
    m_submeshes.compact();

    for (int i = 0; i < m_submeshes.getSize(); i++)
        if (m_isInMemory)
            m_submeshes[i].indices->compact();
}

//------------------------------------------------------------------------

void MeshBase::resetVertices(int num)
{
    FW_ASSERT(num >= 0);
    FW_ASSERT(isInMemory());

    m_vertices.reset(num * m_stride);
    if (num > m_numVertices)
        memset(m_vertices.getPtr(m_numVertices * m_stride), 0, (num - m_numVertices) * m_stride);
    m_numVertices = num;
    freeVBO();
}

//------------------------------------------------------------------------

void MeshBase::resizeVertices(int num)
{
    FW_ASSERT(num >= 0);
    FW_ASSERT(isInMemory());

    m_vertices.resize(num * m_stride);
    if (num > m_numVertices)
        memset(m_vertices.getPtr(m_numVertices * m_stride), 0, (num - m_numVertices) * m_stride);
    m_numVertices = num;
    freeVBO();
}

//------------------------------------------------------------------------

Vec4f MeshBase::getVertexAttrib(int idx, int attrib) const
{
    const AttribSpec& spec = attribSpec(attrib);
    const U8* ptr = vertex(idx) + spec.offset;
    Vec4f v(0.0f, 0.0f, 0.0f, 1.0f);

    for (int i = 0; i < spec.length; i++)
    {
        switch (spec.format)
        {
        case AttribFormat_U8:   v[i] = (F32)ptr[i]; break;
        case AttribFormat_S32:  v[i] = (F32)((S32*)ptr)[i]; break;
        case AttribFormat_F32:  v[i] = ((F32*)ptr)[i]; break;
        default:                FW_ASSERT(false); break;
        }
    }
    return v;
}

//------------------------------------------------------------------------

void MeshBase::setVertexAttrib(int idx, int attrib, const Vec4f& v)
{
    const AttribSpec& spec = attribSpec(attrib);
    U8* ptr = mutableVertex(idx) + spec.offset;

    for (int i = 0; i < spec.length; i++)
    {
        switch (spec.format)
        {
        case AttribFormat_U8:   ptr[i] = (U8)v[i]; break;
        case AttribFormat_S32:  ((S32*)ptr)[i] = (S32)v[i]; break;
        case AttribFormat_F32:  ((F32*)ptr)[i] = v[i]; break;
        default:                FW_ASSERT(false); break;
        }
    }
}

//------------------------------------------------------------------------

void MeshBase::resizeSubmeshes(int num)
{
    FW_ASSERT(isInMemory());
    int old = m_submeshes.getSize();
    if (old == num)
        return;

    for (int i = num; i < old; i++)
    {
        Submesh& sm = m_submeshes[i];
        delete sm.indices;
        for (int j = 0; j < TextureType_Max; j++)
            sm.material.textures[j].clear();
    }

    m_submeshes.resize(num);
    freeVBO();

    for (int i = old; i < num; i++)
    {
        Submesh& sm     = m_submeshes[i];
        sm.indices      = new Array<Vec3i>;
        sm.ofsInVBO     = 0;
        sm.sizeInVBO    = 0;
    }
}

//------------------------------------------------------------------------

Buffer& MeshBase::getVBO(void)
{
    if (m_isInVBO)
        return m_vbo;

    FW_ASSERT(m_isInMemory);
    int ofs = m_vertices.getSize();
    for (int i = 0; i < m_submeshes.getSize(); i++)
    {
        m_submeshes[i].ofsInVBO = ofs;
        m_submeshes[i].sizeInVBO = m_submeshes[i].indices->getSize() * 3;
        ofs += m_submeshes[i].indices->getNumBytes();
    }

    m_vbo.resizeDiscard(ofs);
    memcpy(m_vbo.getMutablePtr(), m_vertices.getPtr(), m_vertices.getSize());
    for (int i = 0; i < m_submeshes.getSize(); i++)
    {
        memcpy(
            m_vbo.getMutablePtr(m_submeshes[i].ofsInVBO),
            m_submeshes[i].indices->getPtr(),
            m_submeshes[i].indices->getNumBytes());
    }

    m_vbo.setOwner(Buffer::GL, false);
    m_vbo.free(Buffer::CPU);
    m_isInVBO = true;
    return m_vbo;
}

//------------------------------------------------------------------------

void MeshBase::setGLAttrib(GLContext* gl, int attrib, int loc)
{
    const AttribSpec& spec = attribSpec(attrib);
    FW_ASSERT(gl);

    GLenum glFormat;
    switch (spec.format)
    {
    case AttribFormat_U8:   glFormat = GL_UNSIGNED_BYTE; break;
    case AttribFormat_S32:  glFormat = GL_INT; break;
    case AttribFormat_F32:  glFormat = GL_FLOAT; break;
    default:                FW_ASSERT(false); return;
    }

    gl->setAttrib(
        loc,
        spec.length,
        glFormat,
        vboAttribStride(attrib),
        getVBO(),
        vboAttribOffset(attrib));
}

//------------------------------------------------------------------------

void MeshBase::draw(GLContext* gl, const Mat4f& posToCamera, const Mat4f& projection, GLContext::Program* prog, bool gouraud)
{
    FW_ASSERT(gl);
    const char* progId = (!gouraud) ? "MeshBase::draw_generic" : "MeshBase::draw_gouraud";
    if (!prog)
        prog = gl->getProgram(progId);

    // Generic shading.

    if (!prog && !gouraud)
    {
        prog = new GLContext::Program(
            "#version 120\n"
            FW_GL_SHADER_SOURCE(
                uniform mat4 posToClip;
                uniform mat4 posToCamera;
                uniform mat3 normalToCamera;
                attribute vec3 positionAttrib;
                attribute vec3 normalAttrib;
                attribute vec4 colorAttrib;
                attribute vec2 texCoordAttrib;
                centroid varying vec3 positionVarying;
                centroid varying vec3 normalVarying;
                centroid varying vec4 colorVarying;
                varying vec2 texCoordVarying;

                void main()
                {
                    vec4 pos = vec4(positionAttrib, 1.0);
                    gl_Position = posToClip * pos;
                    positionVarying = (posToCamera * pos).xyz;
                    normalVarying = normalToCamera * normalAttrib;
                    colorVarying = colorAttrib;
                    texCoordVarying = texCoordAttrib;
                }
            ),
            "#version 120\n"
            FW_GL_SHADER_SOURCE(
                uniform bool hasNormals;
                uniform bool hasDiffuseTexture;
                uniform bool hasAlphaTexture;
                uniform vec4 diffuseUniform;
                uniform vec3 specularUniform;
                uniform float glossiness;
                uniform sampler2D diffuseSampler;
                uniform sampler2D alphaSampler;
                centroid varying vec3 positionVarying;
                centroid varying vec3 normalVarying;
                centroid varying vec4 colorVarying;
                varying vec2 texCoordVarying;

                void main()
                {
                    vec4 diffuseColor = diffuseUniform * colorVarying;
                    vec3 specularColor = specularUniform;

                    if (hasDiffuseTexture)
                        diffuseColor.rgb = texture2D(diffuseSampler, texCoordVarying).rgb;

                    if (hasAlphaTexture)
                        diffuseColor.a = texture2D(alphaSampler, texCoordVarying).g;

                    if (diffuseColor.a <= 0.5)
                        discard;

                    vec3 I = normalize(positionVarying);
                    vec3 N = normalize(normalVarying);
                    float diffuseCoef = (hasNormals) ? max(-dot(I, N), 0.0) * 0.75 + 0.25 : 1.0;
                    float specularCoef = (hasNormals) ? pow(max(-dot(I, reflect(I, N)), 0.0), glossiness) : 0.0;
                    gl_FragColor = vec4(diffuseColor.rgb * diffuseCoef + specularColor * specularCoef, diffuseColor.a);
                }
            ));
    }

    // Gouraud shading.

    if (!prog && gouraud)
    {
        prog = new GLContext::Program(
            "#version 120\n"
            FW_GL_SHADER_SOURCE(
                uniform mat4 posToClip;
                uniform mat4 posToCamera;
                uniform mat3 normalToCamera;
                uniform bool hasNormals;
                uniform vec4 diffuseUniform;
                uniform vec3 specularUniform;
                uniform float glossiness;
                attribute vec3 positionAttrib;
                attribute vec3 normalAttrib;
                attribute vec4 colorAttrib;
                centroid varying vec4 colorVarying;

                void main()
                {
                    vec4 pos = vec4(positionAttrib, 1.0);
                    gl_Position = posToClip * pos;
                    vec3 I = normalize((posToCamera * pos).xyz);
                    vec3 N = normalize(normalToCamera * normalAttrib);
                    float diffuseCoef = (hasNormals) ? max(-dot(I, N), 0.0) * 0.75 + 0.25 : 1.0;
                    float specularCoef = (hasNormals) ? pow(max(-dot(I, reflect(I, N)), 0.0), glossiness) : 0.0;
                    vec4 diffuseColor = diffuseUniform * colorAttrib;
                    colorVarying = vec4(diffuseColor.rgb * diffuseCoef + specularUniform * specularCoef, diffuseColor.a);
                }
            ),
            "#version 120\n"
            FW_GL_SHADER_SOURCE(
                centroid varying vec4 colorVarying;
                void main()
                {
                    gl_FragColor = colorVarying;
                }
            ));
    }

    // Find mesh attributes.

    int posAttrib = findAttrib(AttribType_Position);
    int normalAttrib = findAttrib(AttribType_Normal);
    int colorAttrib = findAttrib(AttribType_Color);
    int texCoordAttrib = findAttrib(AttribType_TexCoord);
    if (posAttrib == -1)
        return;

    // Setup uniforms.

    gl->setProgram(progId, prog);
    prog->use();
    gl->setUniform(prog->getUniformLoc("posToClip"), projection * posToCamera);
    gl->setUniform(prog->getUniformLoc("posToCamera"), posToCamera);
    gl->setUniform(prog->getUniformLoc("normalToCamera"), posToCamera.getXYZ().inverted().transposed());
    gl->setUniform(prog->getUniformLoc("hasNormals"), (normalAttrib != -1));
    gl->setUniform(prog->getUniformLoc("diffuseSampler"), 0);
    gl->setUniform(prog->getUniformLoc("alphaSampler"), 1);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, getVBO().getGLBuffer());
    setGLAttrib(gl, posAttrib, prog->getAttribLoc("positionAttrib"));

    if (normalAttrib != -1)
        setGLAttrib(gl, normalAttrib, prog->getAttribLoc("normalAttrib"));
    else
        glVertexAttrib3f(prog->getAttribLoc("normalAttrib"), 0.0f, 0.0f, 0.0f);

    if (colorAttrib != -1)
        setGLAttrib(gl, colorAttrib, prog->getAttribLoc("colorAttrib"));
    else
        glVertexAttrib4f(prog->getAttribLoc("colorAttrib"), 1.0f, 1.0f, 1.01f, 1.0f);

    if (texCoordAttrib != -1)
        setGLAttrib(gl, texCoordAttrib, prog->getAttribLoc("texCoordAttrib"));
    else
        glVertexAttrib2f(prog->getAttribLoc("texCoordAttrib"), 0.0f, 0.0f);

    // Render each submesh.

    for (int i = 0; i < numSubmeshes(); i++)
    {
        const Material& mat = material(i);
        gl->setUniform(prog->getUniformLoc("diffuseUniform"), mat.diffuse);
        gl->setUniform(prog->getUniformLoc("specularUniform"), mat.specular * 0.5f);
        gl->setUniform(prog->getUniformLoc("glossiness"), mat.glossiness);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, mat.textures[TextureType_Diffuse].getGLTexture());
        gl->setUniform(prog->getUniformLoc("hasDiffuseTexture"), mat.textures[TextureType_Diffuse].exists());

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, mat.textures[TextureType_Alpha].getGLTexture());
        gl->setUniform(prog->getUniformLoc("hasAlphaTexture"), mat.textures[TextureType_Alpha].exists());

        glDrawElements(GL_TRIANGLES, vboIndexSize(i), GL_UNSIGNED_INT, (void*)(UPTR)vboIndexOffset(i));
    }

    gl->resetAttribs();
}

//------------------------------------------------------------------------

void MeshBase::freeMemory(void)
{
    if (!m_isInMemory)
        return;

    m_isInMemory = false;
    m_vertices.reset();
    for (int i = 0; i < m_submeshes.getSize(); i++)
    {
        delete m_submeshes[i].indices;
        m_submeshes[i].indices = NULL;
    }
}

//------------------------------------------------------------------------

void MeshBase::xformPositions(const Mat4f& mat)
{
    int posAttrib = findAttrib(AttribType_Position);
    if (posAttrib == -1)
        return;

    for (int i = 0; i < numVertices(); i++)
    {
        Vec4f pos = getVertexAttrib(i, posAttrib);
        pos = mat * pos;
        if (pos.w != 0.0f)
            pos *= 1.0f / pos.w;
        setVertexAttrib(i, posAttrib, pos);
    }
}

//------------------------------------------------------------------------

void MeshBase::xformNormals(const Mat3f& mat, bool normalize)
{
    int normalAttrib = findAttrib(AttribType_Normal);
    if (normalAttrib == -1)
        return;

    for (int i = 0; i < numVertices(); i++)
    {
        Vec3f normal = getVertexAttrib(i, normalAttrib).getXYZ();
        normal = mat * normal;
        if (normalize)
            normal = normal.normalized();
        setVertexAttrib(i, normalAttrib, Vec4f(normal, 0.0f));
    }
}

//------------------------------------------------------------------------

void MeshBase::getBBox(Vec3f& lo, Vec3f& hi) const
{
    lo = Vec3f(+FW_F32_MAX);
    hi = Vec3f(-FW_F32_MAX);

    int posAttrib = -1;
    for (int i = 0; i < numAttribs(); i++)
        if (attribSpec(i).type == AttribType_Position)
            posAttrib = i;

    if (posAttrib == -1)
        return;

    for (int i = 0; i < numVertices(); i++)
    {
        Vec4f pos = getVertexAttrib(i, posAttrib);
        for (int j = 0; j < 3; j++)
        {
            lo[j] = min(lo[j], pos[j]);
            hi[j] = max(hi[j], pos[j]);
        }
    }
}

//------------------------------------------------------------------------

void MeshBase::recomputeNormals(void)
{
    int posAttrib = findAttrib(AttribType_Position);
    int normalAttrib = findAttrib(AttribType_Normal);
    if (posAttrib == -1 || normalAttrib == -1)
        return;

    // Calculate average normal for each vertex position.

    Hash<Vec3f, Vec3f> posToNormal;
    Vec3f v[3];

    for (int i = 0; i < numSubmeshes(); i++)
    {
        const Array<Vec3i>& tris = indices(i);
        for (int j = 0; j < tris.getSize(); j++)
        {
            const Vec3i& tri = tris[j];
            for (int k = 0; k < 3; k++)
                v[k] = getVertexAttrib(tri[k], posAttrib).getXYZ();

            Vec3f triNormal = (v[1] - v[0]).cross(v[2] - v[0]);
            for (int k = 0; k < 3; k++)
            {
                Vec3f* vertNormal = posToNormal.search(v[k]);
                if (vertNormal)
                    *vertNormal += triNormal;
                else
                    posToNormal.add(v[k], triNormal);
            }
        }
    }

    // Output normals.

    for (int i = 0; i < numVertices(); i++)
    {
        Vec3f pos = getVertexAttrib(i, posAttrib).getXYZ();
        Vec3f* normal = posToNormal.search(pos);
        if (normal)
            setVertexAttrib(i, normalAttrib, Vec4f((*normal).normalized(), 0.0f));
    }
}

//------------------------------------------------------------------------

void MeshBase::flipTriangles(void)
{
    for (int i = 0; i < numSubmeshes(); i++)
    {
        Array<Vec3i>& tris = mutableIndices(i);
        for (int j = 0; j < tris.getSize(); j++)
            swap(tris[j].x, tris[j].y);
    }
}

//------------------------------------------------------------------------

void MeshBase::clean(void)
{
    // Remove degenerate triangles and empty submeshes.

    int submeshOut = 0;
    for (int submeshIn = 0; submeshIn < numSubmeshes(); submeshIn++)
    {
        Array<Vec3i>& inds = mutableIndices(submeshIn);
        int indOut = 0;
        for (int i = 0; i < inds.getSize(); i++)
        {
            const Vec3i& v = inds[i];
            if (v.x != v.y && v.x != v.z && v.y != v.z)
                inds[indOut++] = v;
        }

        if (indOut)
        {
            inds.resize(indOut);
            if (submeshOut != submeshIn)
            {
                material(submeshOut) = material(submeshIn);
                mutableIndices(submeshOut) = inds;
            }
            mutableIndices(submeshOut).compact();
            submeshOut++;
        }
    }
    resizeSubmeshes(submeshOut);

    // Tag referenced vertices.

    Array<U8> vertUsed;
    vertUsed.reset(numVertices());
    memset(vertUsed.getPtr(), 0, vertUsed.getNumBytes());

    for (int submeshIn = 0; submeshIn < numSubmeshes(); submeshIn++)
    {
        const Array<Vec3i>& inds = indices(submeshIn);
        for (int i = 0; i < inds.getSize(); i++)
            for (int j = 0; j < 3; j++)
                vertUsed[inds[i][j]] = 1;
    }

    // Compact vertex array.

    Array<S32> vertRemap;
    vertRemap.reset(numVertices());
    memset(vertRemap.getPtr(), -1, vertRemap.getNumBytes());
    U8* vertPtr = getMutableVertexPtr();
    int vertStride = vertexStride();

    int vertOut = 0;
    for (int vertIn = 0; vertIn < vertUsed.getSize(); vertIn++)
    {
        if (!vertUsed[vertIn])
            continue;

        vertRemap[vertIn] = vertOut;
        if (vertOut != vertIn)
            memcpy(vertPtr + vertOut * vertStride, vertPtr + vertIn * vertStride, vertStride);
        vertOut++;
    }
    resizeVertices(vertOut);

    // Remap indices.

    for (int submeshIdx = 0; submeshIdx < numSubmeshes(); submeshIdx++)
    {
        Array<Vec3i>& inds = mutableIndices(submeshIdx);
        for (int i = 0; i < inds.getSize(); i++)
            for (int j = 0; j < 3; j++)
                inds[i][j] = vertRemap[inds[i][j]];
    }
}

//------------------------------------------------------------------------

void MeshBase::collapseVertices(void)
{
    // Collapse vertices.

    int num = numVertices();
    U8* vertPtr = getMutableVertexPtr();
    int vertStride = vertexStride();

    Hash<GenericHashKey, S32> hash;
    Array<S32> remap;
    hash.setCapacity(num);
    remap.reset(num);

    for (int i = 0; i < num; i++)
    {
        GenericHashKey key(vertPtr + i * vertStride, vertStride);
        S32* found = hash.search(key);
        if (found)
            remap[i] = *found;
        else
        {
            remap[i] = hash.getSize();
            hash.add(key, remap[i]);
            if (remap[i] != i)
                memcpy(vertPtr + remap[i] * vertStride, vertPtr + i * vertStride, vertStride);
        }
    }

    resizeVertices(hash.getSize());

    // Remap indices.

    for (int submeshIdx = 0; submeshIdx < numSubmeshes(); submeshIdx++)
    {
        Array<Vec3i>& inds = mutableIndices(submeshIdx);
        for (int i = 0; i < inds.getSize(); i++)
            for (int j = 0; j < 3; j++)
                inds[i][j] = remap[inds[i][j]];
    }
}

//------------------------------------------------------------------------

void MeshBase::dupVertsPerSubmesh(void)
{
    // Find shared vertices and remap indices.

    int num = numVertices();
    Array<Vec2i> remap;
    Array<S32> dup;
    remap.reset(num);
    memset(remap.getPtr(), -1, remap.getNumBytes());

    for (int submeshIdx = 0; submeshIdx < numSubmeshes(); submeshIdx++)
    {
        Array<Vec3i>& inds = mutableIndices(submeshIdx);
        for (int i = 0; i < inds.getSize(); i++)
        for (int j = 0; j < 3; j++)
        {
            int v = inds[i][j];
            if (remap[v].x != submeshIdx)
            {
                remap[v].x = submeshIdx;
                if (remap[v].y == -1)
                    remap[v].y = v;
                else
                {
                    remap[v].y = num + dup.getSize();
                    dup.add(v);
                }
            }
            inds[i][j] = remap[v].y;
        }
    }

    // Duplicate vertices.

    resizeVertices(num + dup.getSize());
    U8* vertPtr = getMutableVertexPtr();
    int vertStride = vertexStride();

    for (int i = 0; i < dup.getSize(); i++)
        memcpy(vertPtr + (num + i) * vertStride, vertPtr + dup[i] * vertStride, vertStride);
}

//------------------------------------------------------------------------

void MeshBase::fixMaterialColors(void)
{
    for (int submeshIdx = 0; submeshIdx < numSubmeshes(); submeshIdx++)
    {
        Material& mat = material(submeshIdx);
        Texture tex = mat.textures[TextureType_Diffuse];
        if (tex.exists())
        {
            Vec4f avg = tex.getMipLevel(64).getImage()->getVec4f(0);
            mat.diffuse = Vec4f(avg.getXYZ(), mat.diffuse.w);
        }
    }
}

//------------------------------------------------------------------------

void MeshBase::simplify(F32 maxError)
{
    struct Vertex
    {
        Vec3f   pos;
        F32     error;
        F32     posWeight;
        F32     outWeight;
        S32     firstEdge;
        S32     timeStamp;
        S32     outIdx;
    };

    struct Edge
    {
        Vec2i   verts;
        Vec2i   next;
    };

    // Find attributes.

    int posAttrib = findAttrib(MeshBase::AttribType_Position);
    int normalAttrib = findAttrib(MeshBase::AttribType_Normal);
    if (posAttrib == -1)
        return;

    // Group vertices.

    Array<Vertex> verts(NULL, numVertices());
    UnionFind posGroups(numVertices()); // by position
    UnionFind outGroups(numVertices()); // by all attributes
    {
        Hash<Vec3f, S32> posHash;
        Hash<GenericHashKey, S32> outHash;
        for (int i = 0; i < verts.getSize(); i++)
        {
            Vertex& v   = verts[i];
            v.pos       = getVertexAttrib(i, posAttrib).getXYZ();
            v.error     = 0.0f;
            v.posWeight = 0.0f;
            v.outWeight = 0.0f;
            v.firstEdge = -1;
            v.timeStamp = -1;
            v.outIdx    = -1;

            S32* group = posHash.search(v.pos);
            posGroups.unionSets(i, (group) ? *group : posHash.add(v.pos, i));

            GenericHashKey outKey(getVertexPtr(i), vertexStride());
            group = outHash.search(outKey);
            outGroups.unionSets(i, (group) ? *group : outHash.add(outKey, i));
        }
    }

    // Collect edges and accumulate weights.

    Array<Edge> edges;
    {
        Set<Vec2i> edgeSet;
        for (int submeshIdx = 0; submeshIdx < numSubmeshes(); submeshIdx++)
        {
            const Array<Vec3i>& tris = indices(submeshIdx);
            for (int triIdx = 0; triIdx < tris.getSize(); triIdx++)
            {
                const Vec3i& tri = tris[triIdx];
                F32 area = max(length(cross(verts[tri.y].pos - verts[tri.x].pos, verts[tri.z].pos - verts[tri.x].pos)), 1.0e-8f);
                for (int i = 0; i < 3; i++)
                {
                    Vec2i vi = Vec2i(posGroups[tri[i]], posGroups[tri[(i == 2) ? 0 : (i + 1)]]);
                    verts[vi.x].posWeight += area;
                    verts[outGroups[tri[i]]].outWeight += area;

                    if (vi.x > vi.y)
                        swap(vi.x, vi.y);
                    if (vi.x == vi.y || edgeSet.contains(vi))
                        continue;

                    edgeSet.add(vi);
                    int ei = edges.getSize();
                    Edge& e = edges.add();
                    e.verts = vi;

                    for (int j = 0; j < 2; j++)
                    {
                        S32* prevSlot = &verts[vi[j]].firstEdge;
                        if (*prevSlot == -1)
                            e.next[j] = ei;
                        else
                        {
                            prevSlot = &edges[*prevSlot].next[(edges[*prevSlot].verts.x == vi[j]) ? 0 : 1];
                            e.next[j] = *prevSlot;
                        }
                        *prevSlot = ei;
                    }
                }
            }
        }
    }

    // Create binary heap of edges.

    BinaryHeap<F32> edgeHeap;
    for (int i = 0; i < edges.getSize(); i++)
    {
        const Vertex& va = verts[edges[i].verts.x];
        const Vertex& vb = verts[edges[i].verts.y];
        F32 error = max(va.posWeight, vb.posWeight) * length(va.pos - vb.pos) / (va.posWeight + vb.posWeight);
        if (error <= maxError)
            edgeHeap.add(i, error);
    }

    // Collapse edges.

    for (int timeStamp = 0; !edgeHeap.isEmpty(); timeStamp++)
    {
        // Group removed vertices to form a new one.

        int removedEdge = edgeHeap.getMinIndex();
        const Vec2i& removedVerts = edges[removedEdge].verts;
        const Vertex& va = verts[removedVerts.x];
        const Vertex& vb = verts[removedVerts.y];

        int vi      = posGroups.unionSets(removedVerts.x, removedVerts.y);
        Vertex& v   = verts[vi];
        v.pos       = (va.pos * va.posWeight + vb.pos * vb.posWeight) / (va.posWeight + vb.posWeight);
        v.error     = edgeHeap.remove(removedEdge);
        v.posWeight = va.posWeight + vb.posWeight;
        v.firstEdge = -1;

        // Update connected edges.

        for (int i = 0; i < 2; i++)
        {
            int nextEdge = edges[removedEdge].next[i];
            while (nextEdge != removedEdge)
            {
                int currEdge = nextEdge;
                Edge& e = edges[currEdge];
                int c = (e.verts.x == removedVerts[i]) ? 0 : 1;
                nextEdge = e.next[c];
                edgeHeap.remove(currEdge);

                // Duplicate => discard.

                int otherVertex = e.verts[1 - c];
                if (otherVertex == -1 || verts[otherVertex].timeStamp == timeStamp)
                {
                    e.verts[c] = -1;
                    continue;
                }

                // Update the edge.

                Vertex& vo = verts[otherVertex];
                vo.timeStamp = timeStamp;
                e.verts[c] = vi;

                S32* prevSlot = &v.firstEdge;
                if (*prevSlot == -1)
                    e.next[c] = currEdge;
                else
                {
                    prevSlot = &edges[*prevSlot].next[(edges[*prevSlot].verts.x == vi) ? 0 : 1];
                    e.next[c] = *prevSlot;
                }
                *prevSlot = currEdge;

                // Add to the heap.

                F32 coef = length(v.pos - vo.pos) / (v.posWeight + vo.posWeight);
                F32 error = max(v.error + vo.posWeight * coef, vo.error + v.posWeight * coef);
                if (error <= maxError)
                    edgeHeap.add(currEdge, error);
            }
        }
    }

    // For each degenerate edge, remove triangle and group vertices.

    for (int submeshIdx = 0; submeshIdx < numSubmeshes(); submeshIdx++)
    {
        Array<Vec3i>& tris = mutableIndices(submeshIdx);
        for (int triIdx = tris.getSize() - 1; triIdx >= 0; triIdx--)
        {
            const Vec3i& tri = tris[triIdx];
            bool degen = false;
            for (int i = 0; i < 3; i++)
            {
                int j = (i == 2) ? 0 : (i + 1);
                if (posGroups[tri[i]] != posGroups[tri[j]])
                    continue;
                degen = true;
                outGroups.unionSets(tri[i], tri[j]);
            }
            if (degen)
                tris.removeSwap(triIdx);
        }
        tris.compact();
    }

    // Assign output vertices.

    int numOutVerts = 0;
    for (int submeshIdx = 0; submeshIdx < numSubmeshes(); submeshIdx++)
    {
        Array<Vec3i>& tris = mutableIndices(submeshIdx);
        for (int triIdx = 0; triIdx < tris.getSize(); triIdx++)
        {
            Vec3i& tri = tris[triIdx];
            for (int i = 0; i < 3; i++)
            {
                S32& idx = verts[outGroups[tri[i]]].outIdx;
                if (idx == -1)
                    idx = numOutVerts++;
                tri[i] = idx;
            }
        }
    }

    // Initialize output vertices.

    int numAttrib = numAttribs();
    Array<Vec4f> outAttribs(NULL, numOutVerts * numAttrib);
    Array<F32> outDenom(NULL, numOutVerts);
    memset(outAttribs.getPtr(), 0, outAttribs.getNumBytes());
    memset(outDenom.getPtr(), 0, outDenom.getNumBytes());

    // Accumulate vertex attributes.

    for (int i = 0; i < verts.getSize(); i++)
    {
        const Vertex& v = verts[outGroups[i]];
        if (v.outIdx == -1 || v.outWeight == 0.0f)
            continue;

        Vec4f* ptr = outAttribs.getPtr(v.outIdx * numAttrib);
        for (int j = 0; j < numAttrib; j++)
            if (j == posAttrib)
                ptr[j] = Vec4f(verts[posGroups[i]].pos, 1.0f);
            else
                ptr[j] += getVertexAttrib(i, j) * v.outWeight;
        outDenom[v.outIdx] += v.outWeight;
    }

    // Output vertices.

    resetVertices(numOutVerts);
    for (int i = 0; i < numOutVerts; i++)
    {
        const Vec4f* ptr = outAttribs.getPtr(i * numAttrib);
        F32 coef = 1.0f / outDenom[i];
        for (int j = 0; j < numAttrib; j++)
        {
            Vec4f v = ptr[j];
            if (j == normalAttrib)
                v = Vec4f(normalize(v.getXYZ()), 0.0f);
            else if (j != posAttrib)
                v *= coef;
            setVertexAttrib(i, j, v);
        }
    }
}

//------------------------------------------------------------------------

void FW::addCubeToMesh(Mesh<VertexPNC>& mesh, int submesh, const Vec3f& lo, const Vec3f& hi, const Vec4f& color, bool forceNormal, const Vec3f& normal)
{
    VertexPNC vertexArray[] =
    {
        VertexPNC(Vec3f(lo.x, lo.y, hi.z), Vec3f(-1.0f,  0.0f,  0.0f), color),
        VertexPNC(Vec3f(lo.x, hi.y, lo.z), Vec3f(-1.0f,  0.0f,  0.0f), color),
        VertexPNC(Vec3f(lo.x, hi.y, hi.z), Vec3f(-1.0f,  0.0f,  0.0f), color),
        VertexPNC(Vec3f(lo.x, lo.y, lo.z), Vec3f(-1.0f,  0.0f,  0.0f), color),

        VertexPNC(Vec3f(hi.x, lo.y, hi.z), Vec3f(+1.0f,  0.0f,  0.0f), color),
        VertexPNC(Vec3f(hi.x, hi.y, lo.z), Vec3f(+1.0f,  0.0f,  0.0f), color),
        VertexPNC(Vec3f(hi.x, hi.y, hi.z), Vec3f(+1.0f,  0.0f,  0.0f), color),
        VertexPNC(Vec3f(hi.x, lo.y, lo.z), Vec3f(+1.0f,  0.0f,  0.0f), color),

        VertexPNC(Vec3f(lo.x, lo.y, hi.z), Vec3f( 0.0f, -1.0f,  0.0f), color),
        VertexPNC(Vec3f(hi.x, lo.y, lo.z), Vec3f( 0.0f, -1.0f,  0.0f), color),
        VertexPNC(Vec3f(hi.x, lo.y, hi.z), Vec3f( 0.0f, -1.0f,  0.0f), color),
        VertexPNC(Vec3f(lo.x, lo.y, lo.z), Vec3f( 0.0f, -1.0f,  0.0f), color),

        VertexPNC(Vec3f(lo.x, hi.y, hi.z), Vec3f( 0.0f, +1.0f,  0.0f), color),
        VertexPNC(Vec3f(hi.x, hi.y, lo.z), Vec3f( 0.0f, +1.0f,  0.0f), color),
        VertexPNC(Vec3f(hi.x, hi.y, hi.z), Vec3f( 0.0f, +1.0f,  0.0f), color),
        VertexPNC(Vec3f(lo.x, hi.y, lo.z), Vec3f( 0.0f, +1.0f,  0.0f), color),

        VertexPNC(Vec3f(lo.x, hi.y, lo.z), Vec3f( 0.0f,  0.0f, -1.0f), color),
        VertexPNC(Vec3f(hi.x, lo.y, lo.z), Vec3f( 0.0f,  0.0f, -1.0f), color),
        VertexPNC(Vec3f(hi.x, hi.y, lo.z), Vec3f( 0.0f,  0.0f, -1.0f), color),
        VertexPNC(Vec3f(lo.x, lo.y, lo.z), Vec3f( 0.0f,  0.0f, -1.0f), color),

        VertexPNC(Vec3f(lo.x, hi.y, hi.z), Vec3f( 0.0f,  0.0f, +1.0f), color),
        VertexPNC(Vec3f(hi.x, lo.y, hi.z), Vec3f( 0.0f,  0.0f, +1.0f), color),
        VertexPNC(Vec3f(hi.x, hi.y, hi.z), Vec3f( 0.0f,  0.0f, +1.0f), color),
        VertexPNC(Vec3f(lo.x, lo.y, hi.z), Vec3f( 0.0f,  0.0f, +1.0f), color)
    };

    static const Vec3i indexArray[] =
    {
        Vec3i(0,  1,  3),  Vec3i(2,  1,  0),
        Vec3i(6,  4,  5),  Vec3i(5,  4,  7),
        Vec3i(8,  11, 9),  Vec3i(10, 8,  9),
        Vec3i(15, 12, 13), Vec3i(12, 14, 13),
        Vec3i(16, 17, 19), Vec3i(18, 17, 16),
        Vec3i(23, 21, 20), Vec3i(20, 21, 22)
    };

    int base = mesh.numVertices();
    VertexPNC* vertexPtr = mesh.addVertices(vertexArray, FW_ARRAY_SIZE(vertexArray));
    if (forceNormal)
        for (int i = 0; i < (int)FW_ARRAY_SIZE(vertexArray); i++)
            vertexPtr[i].n = normal;

    Array<Vec3i>& indices = mesh.mutableIndices(submesh);
    for (int i = 0; i < (int)FW_ARRAY_SIZE(indexArray); i++)
        indices.add(indexArray[i] + base);
}

//------------------------------------------------------------------------

MeshBase* FW::importMesh(const String& fileName)
{
    String lower = fileName.toLower();

#define STREAM(CALL) { File file(fileName, File::Read); BufferedInputStream stream(file); return CALL; }
    if (lower.endsWith(".bin")) STREAM(importBinaryMesh(stream))
    if (lower.endsWith(".obj")) STREAM(importWavefrontMesh(stream, fileName))
#undef STREAM

    setError("importMesh(): Unsupported file extension '%s'!", fileName.getPtr());
    return NULL;
}

//------------------------------------------------------------------------

void FW::exportMesh(const String& fileName, const MeshBase* mesh)
{
    FW_ASSERT(mesh);
    String lower = fileName.toLower();

#define STREAM(CALL) { File file(fileName, File::Create); BufferedOutputStream stream(file); CALL; stream.flush(); return; }
    if (lower.endsWith(".bin")) STREAM(exportBinaryMesh(stream, mesh))
    if (lower.endsWith(".obj")) STREAM(exportWavefrontMesh(stream, mesh, fileName))
#undef STREAM

    setError("exportMesh(): Unsupported file extension '%s'!", fileName.getPtr());
}

//------------------------------------------------------------------------

String FW::getMeshImportFilter(void)
{
    return
        "obj:Wavefront Mesh,"
        "bin:Binary Mesh";
}

//------------------------------------------------------------------------

String FW::getMeshExportFilter(void)
{
    return
        "obj:Wavefront Mesh,"
        "bin:Binary Mesh";
}

//------------------------------------------------------------------------
