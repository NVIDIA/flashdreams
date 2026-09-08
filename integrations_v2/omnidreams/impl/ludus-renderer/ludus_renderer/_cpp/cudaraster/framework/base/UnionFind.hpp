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
#include "base/Array.hpp"

namespace FW
{
//------------------------------------------------------------------------

class UnionFind
{
public:
    explicit            UnionFind   (int capacity = 0)          { setCapacity(capacity); }
                        UnionFind   (const UnionFind& other)    { set(other); }
                        ~UnionFind  (void)                      {}

    int                 unionSets   (int idxA, int idxB);
    int                 findSet     (int idx) const;
    bool                isSameSet   (int idxA, int idxB) const  { return (findSet(idxA) == findSet(idxB)); }

    void                clear       (void)                      { m_sets.clear(); }
    void                reset       (void)                      { m_sets.reset(); }
    void                setCapacity (int capacity)              { m_sets.setCapacity(capacity); }
    void                set         (const UnionFind& other)    { m_sets = other.m_sets; }

    UnionFind&          operator=   (const UnionFind& other)    { set(other); return *this; }
    int                 operator[]  (int idx) const             { return findSet(idx); }

private:
    mutable Array<S32>  m_sets;
};

//------------------------------------------------------------------------
}
