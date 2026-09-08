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

#include "base/UnionFind.hpp"

using namespace FW;

//------------------------------------------------------------------------

int UnionFind::unionSets(int idxA, int idxB)
{
    // Grow the array.

    FW_ASSERT(idxA >= 0 && idxB >= 0);
    int oldSize = m_sets.getSize();
    int newSize = max(idxA, idxB) + 1;
    if (newSize > oldSize)
    {
        m_sets.resize(newSize);
        for (int i = oldSize; i < newSize; i++)
            m_sets[i] = i;
    }

    // Union the sets.

    int root = findSet(idxA);
    m_sets[findSet(idxB)] = root;
    return root;
}

//------------------------------------------------------------------------

int UnionFind::findSet(int idx) const
{
    // Out of the array => isolated.

    if (idx < 0 || idx >= m_sets.getSize())
        return idx;

    // Find the root set.

    int root = idx;
    for (;;)
    {
        int parent = m_sets[root];
        if (parent == root)
            break;
        root = parent;
    }

    // Update parent links to point directly to the root.

    for (;;)
    {
        int parent = m_sets[idx];
        if (parent == root)
            break;
        m_sets[idx] = root;
        idx = parent;
    }
    return root;
}

//------------------------------------------------------------------------
