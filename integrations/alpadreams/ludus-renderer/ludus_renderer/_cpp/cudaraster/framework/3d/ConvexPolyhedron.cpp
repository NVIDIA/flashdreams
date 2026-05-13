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

#include "3d/ConvexPolyhedron.hpp"

using namespace FW;

//------------------------------------------------------------------------

#define EPSILON         1.0e-6f
#define ENABLE_ASSERTS  0

//------------------------------------------------------------------------

void ConvexPolyhedron::setCube(const Vec3f& lo, const Vec3f& hi)
{
    static const Vec2i edges[] =
    {
        Vec2i(0, 1), Vec2i(0, 2), Vec2i(0, 4), Vec2i(1, 3),
        Vec2i(1, 5), Vec2i(2, 3), Vec2i(2, 6), Vec2i(3, 7),
        Vec2i(4, 5), Vec2i(4, 6), Vec2i(5, 7), Vec2i(6, 7),
    };

    static const Vec4i faceEdges[] =
    {
        Vec4i(2, 9, ~6, ~1), Vec4i(3, 7,  ~10, ~4),
        Vec4i(0, 4, ~8, ~2), Vec4i(6, 11, ~7,  ~5),
        Vec4i(1, 5, ~3, ~0), Vec4i(8, 10, ~11, ~9),
    };

    m_vertices.resize(8);
    m_vertices[0].pos = Vec3f(lo.x, lo.y, lo.z);
    m_vertices[1].pos = Vec3f(hi.x, lo.y, lo.z);
    m_vertices[2].pos = Vec3f(lo.x, hi.y, lo.z);
    m_vertices[3].pos = Vec3f(hi.x, hi.y, lo.z);
    m_vertices[4].pos = Vec3f(lo.x, lo.y, hi.z);
    m_vertices[5].pos = Vec3f(hi.x, lo.y, hi.z);
    m_vertices[6].pos = Vec3f(lo.x, hi.y, hi.z);
    m_vertices[7].pos = Vec3f(hi.x, hi.y, hi.z);

    m_edges.resize(12);
    for (int i = 0; i < 12; i++)
        m_edges[i].verts = edges[i];

    m_faces.resize(6);
    m_faceEdges.resize(24);
    FaceEdge* faceEdgePtr = m_faceEdges.getPtr();
    for (int i = 0; i < 6; i++)
    {
        Face& f             = m_faces[i];
        f.planeEq           = 0.0f;
        f.planeEq[i >> 1]   = ((i & 1) == 0) ? -1.0f : 1.0f;
        f.planeEq.w         = ((i & 1) == 0) ? -lo[i >> 1] : -hi[i >> 1];
        f.planeID           = -1;
        f.firstEdge         = i * 4;
        f.numEdges          = 4;

        for (int j = 0; j < 4; j++)
            (faceEdgePtr++)->edge = faceEdges[i][j];
    }
}

//------------------------------------------------------------------------

bool ConvexPolyhedron::intersect(const Vec4f& planeEq, int planeID)
{
    // Plane matches a face => not intersected.

    for (int i = m_faces.getSize() - 1; i >= 0; i--)
        if (planeEq == m_faces[i].planeEq)
            return false;

    // Compute the orientation of vertices.

    bool culled = false;
    for (int i = m_vertices.getSize() - 1; i >= 0; i--)
    {
        Vertex& v = m_vertices[i];
        F32 t = planeEq.x * v.pos.x + planeEq.y * v.pos.y + planeEq.z * v.pos.z;
        F32 eps = (abs(t) + abs(planeEq.w)) * EPSILON;
        t += planeEq.w;
        v.aux.orient = t + eps;
        culled = (culled || t - eps > 0.0f);
    }

    // No vertices culled => not intersected.

    if (!culled)
        return false;

    // Intersect edges.

    int edgeOfs = 0;
    int firstNewVertex = m_vertices.getSize();
    for (int i = 0; i < m_edges.getSize(); i++)
    {
        // Both vertices are behind the plane => cull.

        Edge& oldEdge = m_edges[i];
        const Vertex& vx = m_vertices[oldEdge.verts.x];
        const Vertex& vy = m_vertices[oldEdge.verts.y];
        if (vx.aux.orient >= 0.0f && vy.aux.orient >= 0.0f)
        {
            oldEdge.remap = -1;
            continue;
        }

        // Remap.

        oldEdge.remap = edgeOfs;
        Edge& newEdge = m_edges[edgeOfs++];
        newEdge.verts = oldEdge.verts;

        // Vertices are on the different side of the plane => clip.

        if (vx.aux.orient >= 0.0f || vy.aux.orient >= 0.0f)
        {
            F32 t = vx.aux.orient / (vx.aux.orient - vy.aux.orient);
            Vec3f pos = lerp(vx.pos, vy.pos, fastClamp(t, 0.0f, 1.0f));

            newEdge.verts[(vx.aux.orient >= 0.0f) ? 0 : 1] = m_vertices.getSize();
            Vertex& v = m_vertices.add();
            v.pos = pos;
        }
    }

    // Intersect faces.

    int firstNewFaceEdge = m_faceEdges.getSize();
    for (int i = m_faces.getSize() - 1; i >= 0; i--)
    {
        Face& face = m_faces[i];
        FaceEdge* faceEdges = m_faceEdges.getPtr(face.firstEdge);
        Vec2i newEdge = -1;

        for (int j = face.numEdges - 1; j >= 0; j--)
        {
            // Edge was culled => remove from the face.

            int oldIdx = faceEdges[j].edge;
            int mask = oldIdx >> 31;
            int newIdx = m_edges[oldIdx ^ mask].remap;
            if (newIdx == -1)
            {
                faceEdges[j].old = faceEdges[--face.numEdges].old;
                continue;
            }

            // Remap and record intersections.

            faceEdges[j].old = newIdx ^ mask;
            const Vec2i& verts = m_edges[newIdx].verts;
            if (verts.x >= firstNewVertex)
            {
                FW_ASSERT(!ENABLE_ASSERTS || newEdge[mask & 1] == -1);
                newEdge[mask & 1] = verts.x;
            }
            else if (verts.y >= firstNewVertex)
            {
                FW_ASSERT(!ENABLE_ASSERTS || newEdge[~mask & 1] == -1);
                newEdge[~mask & 1] = verts.y;
            }
        }

        // No edges => remove face.
        // Intersected => add new edge.

        if (newEdge.x == -1 || newEdge.y == -1)
        {
            FW_ASSERT(!ENABLE_ASSERTS || (newEdge.x == -1 && newEdge.y == -1));
            face.newEdge = -1;
            if (face.numEdges < 3)
                m_faces.removeSwap(i);
        }
        else
        {
            face.newEdge = edgeOfs;
            m_faceEdges.add().old = edgeOfs;
            if (edgeOfs == m_edges.getSize())
                m_edges.add();
            m_edges[edgeOfs++].verts = newEdge;
        }
    }
    m_edges.resize(edgeOfs);

    // Add the new face.

    Face& newFace       = m_faces.add();
    newFace.planeEq     = planeEq;
    newFace.planeID     = planeID;
    newFace.firstEdge   = firstNewFaceEdge;
    newFace.numEdges    = m_faceEdges.getSize() - firstNewFaceEdge;
    newFace.newEdge     = -1;

    // Rebuild the face edge array.

    m_faceEdges.add(NULL, newFace.numEdges);
    int faceEdgeOfs = 0;
    for (int i = m_faces.getSize() - 1; i >= 0; i--)
    {
        Face& face = m_faces[i];
        const FaceEdge* old = m_faceEdges.getPtr(face.firstEdge);
        face.firstEdge = faceEdgeOfs;
        for (int j = face.numEdges - 1; j >= 0; j--)
            m_faceEdges[faceEdgeOfs++].edge = old[j].old;
        if (face.newEdge != -1)
            m_faceEdges[faceEdgeOfs++].edge = ~face.newEdge;
        face.numEdges = faceEdgeOfs - face.firstEdge;
    }
    m_faceEdges.resize(faceEdgeOfs);

    // Remap vertices.

    int vertexOfs = 0;
    for (int i = 0; i < m_vertices.getSize(); i++)
    {
        Vertex& v = m_vertices[i];
        if (i < firstNewVertex && v.aux.orient >= 0.0f)
            continue;

        v.aux.remap = vertexOfs;
        m_vertices[vertexOfs++].pos = v.pos;
    }
    for (int i = m_edges.getSize() - 1; i >= 0; i--)
    {
        Vec2i& v = m_edges[i].verts;
        v.x = m_vertices[v.x].aux.remap;
        v.y = m_vertices[v.y].aux.remap;
    }
    m_vertices.resize(vertexOfs);
    return true;
}

//------------------------------------------------------------------------

bool ConvexPolyhedron::intersect(const ConvexPolyhedron& other)
{
    if (&other == this)
        return false;

    bool intersected = false;
    for (int i = 0; i < other.m_faces.getSize(); i++)
        if (intersect(other.m_faces[i].planeEq, other.m_faces[i].planeID))
            intersected = true;
    return intersected;
}

//------------------------------------------------------------------------

F32 ConvexPolyhedron::computeVolume(void) const
{
    if (!m_faces.getSize())
        return 0.0f;

    F32 volume = 0.0f;
    const Vec3f& base = m_vertices[0].pos;

    for (int i = 0; i < m_faces.getSize(); i++)
    {
        const Face& face = m_faces[i];
        const FaceEdge* faceEdges = m_faceEdges.getPtr(face.firstEdge);
        Vec3f first = getEdgeStartPos(faceEdges[0].edge) - base;

        for (int j = 1; j < face.numEdges; j++)
        {
            Vec2i v = getEdge(faceEdges[j].edge);
            volume += cross(m_vertices[v.x].pos - base, m_vertices[v.y].pos - base).dot(first);
        }
    }
    return volume * (1.0f / 6.0f);
}

//------------------------------------------------------------------------

F32 ConvexPolyhedron::computeArea(int faceIdx) const
{
    const Face& face = m_faces[faceIdx];
    const FaceEdge* faceEdges = m_faceEdges.getPtr(face.firstEdge);
    const Vec3f& base = getEdgeStartPos(faceEdges[0].edge);
    F32 area = 0.0f;

    for (int i = 1; i < face.numEdges; i++)
    {
        Vec2i v = getEdge(faceEdges[i].edge);
        area += cross(m_vertices[v.x].pos - base, m_vertices[v.y].pos - base).length();
    }
    return area * 0.5f;
}

//------------------------------------------------------------------------

Vec3f ConvexPolyhedron::computeCenterOfMass(int faceIdx) const
{
    const Face& face = m_faces[faceIdx];
    const FaceEdge* faceEdges = m_faceEdges.getPtr(face.firstEdge);
    const Vec3f& base = getEdgeStartPos(faceEdges[0].edge);
    Vec3f pos = 0.0f;
    F32 area = 0.0f;

    for (int i = 1; i < face.numEdges; i++)
    {
        Vec2i v = getEdge(faceEdges[i].edge);
        const Vec3f& px = m_vertices[v.x].pos;
        const Vec3f& py = m_vertices[v.y].pos;
        F32 t = cross(px - base, py - base).length();
        pos += (px + py) * t;
        area += t;
    }
    return pos * (1.0f / 3.0f / area) + base * (1.0f / 3.0f);
}

//------------------------------------------------------------------------

Vec3f ConvexPolyhedron::computeCenterOfMass(void) const
{
    if (!m_faces.getSize())
        return 0.0f;

    Vec3f pos = 0.0f;
    F32 volume = 0.0f;
    const Vec3f& base = m_vertices[0].pos;

    for (int i = 0; i < m_faces.getSize(); i++)
    {
        const Face& face = m_faces[i];
        const FaceEdge* faceEdges = m_faceEdges.getPtr(face.firstEdge);
        Vec3f first = getEdgeStartPos(faceEdges[0].edge) - base;

        for (int j = 1; j < face.numEdges; j++)
        {
            Vec2i v = getEdge(faceEdges[j].edge);
            const Vec3f& px = m_vertices[v.x].pos;
            const Vec3f& py = m_vertices[v.y].pos;
            F32 t = cross(px - base, py - base).dot(first);
            pos += (px + py + first) * t;
            volume += t;
        }
    }
    return pos * (1.0f / 4.0f / volume) + base * (1.0f / 2.0f);
}

//------------------------------------------------------------------------

Mesh<VertexPN>* ConvexPolyhedron::createMesh(void) const
{
    Mesh<VertexPN>* mesh = new Mesh<VertexPN>;
    Array<Vec3i>& inds = mesh->mutableIndices(mesh->addSubmesh());
    Array<S32> vmap(NULL, m_vertices.getSize());

    for (int i = m_faces.getSize() - 1; i >= 0; i--)
    {
        const Face& face = m_faces[i];
        const FaceEdge* faceEdges = m_faceEdges.getPtr(face.firstEdge);
        int base = -1;

        for (int j = face.numEdges - 1; j >= 0; j--)
        {
            base = getEdgeStartVertex(faceEdges[j].edge);
            vmap[base] = mesh->numVertices();
            VertexPN& v = mesh->addVertex();
            v.p = m_vertices[base].pos;
            v.n = face.planeEq.getXYZ();
        }

        for (int j = face.numEdges - 1; j >= 0; j--)
        {
            Vec2i e = getEdge(faceEdges[j].edge);
            if (e.x != base && e.y != base)
                inds.add(Vec3i(vmap[base], vmap[e.x], vmap[e.y]));
        }
    }
    return mesh;
}

//------------------------------------------------------------------------
