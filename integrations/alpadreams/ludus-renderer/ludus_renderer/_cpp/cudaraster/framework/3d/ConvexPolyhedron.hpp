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
#include "3d/Mesh.hpp"

namespace FW
{
//------------------------------------------------------------------------

class ConvexPolyhedron
{
public:
    struct Vertex
    {
        Vec3f           pos;
        union
        {
            F32         orient;
            S32         remap;
        }               aux;
    };

    struct Edge
    {
        Vec2i           verts;
        S32             remap;
    };

    struct Face
    {
        Vec4f           planeEq;
        S32             planeID;
        S32             firstEdge;
        S32             numEdges;
        S32             newEdge;
    };

    struct FaceEdge
    {
        S32             edge;
        S32             old;
    };

public:
                        ConvexPolyhedron    (void)                              {}
                        ConvexPolyhedron    (const Vec3f& lo, const Vec3f& hi)  { setCube(lo, hi); }
                        ConvexPolyhedron    (const ConvexPolyhedron& other)     { set(other); }
                        ~ConvexPolyhedron   (void)                              {}

    int                 getNumVertices      (void) const                        { return m_vertices.getSize(); }
    const Vec3f&        getVertex           (int idx) const                     { return m_vertices[idx].pos; }

    int                 getNumEdges         (void) const                        { return m_edges.getSize(); }
    Vec2i               getEdge             (int idx) const                     { const Vec2i& e = m_edges[idx ^ (idx >> 31)].verts; return (idx >= 0) ? e : Vec2i(e.y, e.x); }
    int                 getEdgeStartVertex  (int idx) const                     { return (idx >= 0) ? m_edges[idx].verts.x : m_edges[~idx].verts.y; }
    int                 getEdgeEndVertex    (int idx) const                     { return (idx >= 0) ? m_edges[idx].verts.y : m_edges[~idx].verts.x; }
    const Vec3f&        getEdgeStartPos     (int idx) const                     { return m_vertices[getEdgeStartVertex(idx)].pos; }
    const Vec3f&        getEdgeEndPos       (int idx) const                     { return m_vertices[getEdgeEndVertex(idx)].pos; }

    int                 getNumFaces         (void) const                        { return m_faces.getSize(); }
    const Vec4f&        getFacePlaneEq      (int idx) const                     { return m_faces[idx].planeEq; }
    int                 getFacePlaneID      (int idx) const                     { return m_faces[idx].planeID; }
    int                 getFaceNumEdges     (int idx) const                     { return m_faces[idx].numEdges; }
    int                 getFaceEdge         (int faceIdx, int idx)              { FW_ASSERT(idx >= 0 && idx < m_faces[faceIdx].numEdges); return m_faceEdges[m_faces[faceIdx].firstEdge + idx].edge; }

    void                set                 (const ConvexPolyhedron& other)     { m_vertices = other.m_vertices; m_edges = other.m_edges; m_faces = other.m_faces; m_faceEdges = other.m_faceEdges; }
    void                setEmpty            (void)                              { m_vertices.clear(); m_edges.clear(); m_faces.clear(); m_faceEdges.clear(); }
    void                setCube             (const Vec3f& lo, const Vec3f& hi);

    bool                intersect           (const Vec4f& planeEq, int planeID = -1);
    bool                intersect           (const ConvexPolyhedron& other);
    bool                intersectCube       (const Vec3f& lo, const Vec3f& hi)  { ConvexPolyhedron t; t.setCube(lo, hi); return intersect(t); }

    F32                 computeVolume       (void) const;
    F32                 computeArea         (int faceIdx) const;
    F32                 computeArea         (void) const                        { F32 t = 0.0f; for (int i = 0; i < m_faces.getSize(); i++) t += computeArea(i); return t; }
    Vec3f               computeCenterOfMass (int faceIdx) const;
    Vec3f               computeCenterOfMass (void) const;
    Mesh<VertexPN>*     createMesh          (void) const;

    ConvexPolyhedron&   operator=           (const ConvexPolyhedron& other)     { set(other); return *this; }

private:
    Array<Vertex>       m_vertices;
    Array<Edge>         m_edges;
    Array<Face>         m_faces;
    Array<FaceEdge>     m_faceEdges;
};

//------------------------------------------------------------------------
}
