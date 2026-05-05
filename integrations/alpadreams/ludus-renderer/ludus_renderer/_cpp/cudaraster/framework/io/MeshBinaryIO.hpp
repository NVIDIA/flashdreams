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
#include "base/Defs.hpp"

namespace FW
{
//------------------------------------------------------------------------

class MeshBase;
class InputStream;
class OutputStream;

//------------------------------------------------------------------------

MeshBase*   importBinaryMesh    (InputStream& stream);
void        exportBinaryMesh    (OutputStream& stream, const MeshBase* mesh);

//------------------------------------------------------------------------
/*

Binary mesh file format v4
--------------------------

- the basic units of data are 32-bit little-endian ints and floats

BinaryMesh
    0       6       struct  v1  MeshHeader
    6       n*3     struct  v1  array of AttribSpec (MeshHeader.numAttribs)
    ?       n*?     struct  v1  array of Vertex (MeshHeader.numVertices)
    ?       n*?     struct  v2  array of Texture (MeshHeader.numTextures)
    ?       n*?     struct  v1  array of Submesh (MeshHeader.numSubmeshes)
    ?

MeshHeader
    0       2       bytes   v1  formatID (must be "BinMesh ")
    2       1       int     v1  formatVersion (must be 4)
    3       1       int     v1  numAttribs
    4       1       int     v1  numVertices
    5       1       int     v2  numTextures
    6       1       int     v1  numSubmeshes
    7

AttribSpec
    0       1       int     v1  type (see MeshBase::AttribType)
    1       1       int     v1  format (see MeshBase::AttribFormat)
    2       1       int     v1  length
    3

Vertex
    0       ?       bytes   v1  array of values (dictated by the set of AttribSpecs)
    ?

Texture
    0       1       int     v2  idLength
    1       ?       bytes   v2  idString
    ?       ?       struct  v2  BinaryImage (see Image.hpp)
    ?

Submesh
    0       3       float   v1  ambient (ignored)
    3       4       float   v1  diffuse
    7       3       float   v1  specular
    10      1       float   v1  glossiness
    11      1       float   v3  displacementCoef
    12      1       float   v3  displacementBias
    13      1       int     v2  diffuseTexture (-1 if none)
    14      1       int     v2  alphaTexture (-1 if none)
    15      1       int     v3  displacementTexture (-1 if none)
    16      1       int     v4  normalTexture (-1 if none)
    17      1       int     v4  environmentTexture (-1 if none)
    18      1       int     v1  numTriangles
    19      n*3     int     v1  indices
    ?

*/
//------------------------------------------------------------------------
}
