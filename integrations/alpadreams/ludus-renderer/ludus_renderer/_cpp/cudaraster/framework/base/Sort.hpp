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
// Generic quick sort implementation.
// Somewhat cumbersome to use - see below for convenience wrappers.
//
// Sort integers into ascending order:
//
//   static bool myCompareFunc   (void* data, int idxA, int idxB)    { return (((S32*)data)[idxA] < ((S32*)data)[idxB]); }
//   static void mySwapFunc      (void* data, int idxA, int idxB)    { swap(((S32*)data)[idxA], ((S32*)data)[idxB]); }
//
//   Array<S32> myArray = ...;
//   sort(myArray.getPtr(), 0, myArray.getSize(), myCompareFunc, mySwapFunc);
//------------------------------------------------------------------------

typedef bool    (*SortCompareFunc) (void* data, int idxA, int idxB);    // Returns true if A should come before B.
typedef void    (*SortSwapFunc)    (void* data, int idxA, int idxB);    // Swaps A and B.

void sort(void* data, int start, int end, SortCompareFunc compareFunc, SortSwapFunc swapFunc, bool multicore = false);

//------------------------------------------------------------------------
// Template-based wrappers.
// Use these if your type defines operator< and you want to sort in
// ascending order, OR if you want to explicitly define a custom
// comparator function.
//
// Sort integers into ascending order:
//
//   Array<S32> myArray = ...;
//   sort(myArray);                             // sort the full array
//   sort(myArray, 0, myArray.getSize());       // specify range of elements
//   sort(myArray.getPtr(), myArray.getSize()); // sort C array
//
// Sort integers into descending order:
//
//   static bool myCompareFunc(void* data, int idxA, int idxB)
//   {
//       const S32& a = ((S32*)data)[idxA];
//       const S32& b = ((S32*)data)[idxB];
//       return (a > b);
//   }
//
//   Array<S32> myArray = ...;
//   sort(myArray, myCompareFunc);
//------------------------------------------------------------------------

template <class T> bool sortDefaultCompare  (void* data, int idxA, int idxB);
template <class T> void sortDefaultSwap     (void* data, int idxA, int idxB);

template <class T> void sort(T* data, int num, SortCompareFunc compareFunc = sortDefaultCompare<T>, SortSwapFunc swapFunc = sortDefaultSwap<T>, bool multicore = false);
template <class T> void sort(Array<T>& data, SortCompareFunc compareFunc = sortDefaultCompare<T>, SortSwapFunc swapFunc = sortDefaultSwap<T>, bool multicore = false);
template <class T> void sort(Array<T>& data, int start, int end, SortCompareFunc compareFunc = sortDefaultCompare<T>, SortSwapFunc swapFunc = sortDefaultSwap<T>, bool multicore = false);

//------------------------------------------------------------------------
// Macro-based wrappers.
// Use these if you want to use a custom comparator, but do not feel like
// defining a separate function for it.
//
// Sort integers into ascending order:
//
//   Array<S32> myArray = ...;
//   FW_SORT_ARRAY(myArray, S32, a < b);                            // sort the full array
//   FW_SORT_SUBARRAY(myArray, 0, myArray.getSize(), S32, a < b);   // specify range of elements
//   FW_SORT(myArray.getPtr(), myArray.getSize(), S32, a < b);      // sort C array
//
// Sort vectors in descending order based on their first component:
//
//   Array<Vec2i> myArray = ...;
//   FW_SORT_ARRAY(myArray, Vec2i, a.x > b.x);
//------------------------------------------------------------------------

#define FW_SORT(PTR, NUM, TYPE, COMPARE)                                FW_SORT_IMPL(PTR, NUM, TYPE, COMPARE, false)
#define FW_SORT_ARRAY(ARRAY, TYPE, COMPARE)                             FW_SORT_IMPL(ARRAY.getPtr(), ARRAY.getSize(), TYPE, COMPARE, false)
#define FW_SORT_SUBARRAY(ARRAY, START, END, TYPE, COMPARE)              FW_SORT_IMPL(ARRAY.getPtr(START), (END) - (START), TYPE, COMPARE, false)

#define FW_SORT_MULTICORE(PTR, NUM, TYPE, COMPARE)                      FW_SORT_IMPL(PTR, NUM, TYPE, COMPARE, true)
#define FW_SORT_ARRAY_MULTICORE(ARRAY, TYPE, COMPARE)                   FW_SORT_IMPL(ARRAY.getPtr(), ARRAY.getSize(), TYPE, COMPARE, true)
#define FW_SORT_SUBARRAY_MULTICORE(ARRAY, START, END, TYPE, COMPARE)    FW_SORT_IMPL(ARRAY.getPtr(START), (END) - (START), TYPE, COMPARE, true)

//------------------------------------------------------------------------
// Wrapper implementation.
//------------------------------------------------------------------------

template <class T> bool sortDefaultCompare(void* data, int idxA, int idxB)
{
    return (((T*)data)[idxA] < ((T*)data)[idxB]);
}

//------------------------------------------------------------------------

template <class T> void sortDefaultSwap(void* data, int idxA, int idxB)
{
    swap(((T*)data)[idxA], ((T*)data)[idxB]);
}

//------------------------------------------------------------------------

template <class T> void sort(Array<T>& data, SortCompareFunc compareFunc, SortSwapFunc swapFunc, bool multicore)
{
    sort(data.getPtr(), 0, data.getSize(), compareFunc, swapFunc, multicore);
}

//------------------------------------------------------------------------

template <class T> void sort(Array<T>& data, int start, int end, SortCompareFunc compareFunc, SortSwapFunc swapFunc, bool multicore)
{
    sort(data.getPtr(start), 0, end - start, compareFunc, swapFunc, multicore);
}

//------------------------------------------------------------------------

template <class T> void sort(T* data, int num, SortCompareFunc compareFunc, SortSwapFunc swapFunc, bool multicore)
{
    sort(data, 0, num, compareFunc, swapFunc, multicore);
}

//------------------------------------------------------------------------

#define FW_SORT_IMPL(PTR, NUM, TYPE, COMPARE, MULTICORE) \
    { \
        struct SortLambda \
        { \
            static bool compareFunc(void* data, int idxA, int idxB) \
            { \
                TYPE& a = ((TYPE*)data)[idxA]; \
                TYPE& b = ((TYPE*)data)[idxB]; \
                return COMPARE; \
            } \
        }; \
        FW::sort(PTR, NUM, SortLambda::compareFunc, FW::sortDefaultSwap<TYPE>, MULTICORE); \
    }

//------------------------------------------------------------------------
}
