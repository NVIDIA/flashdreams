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
#include "base/Math.hpp"

namespace FW
{
//------------------------------------------------------------------------
// Growing Array, similar to stl::vector.
// Array<T> uses S32 as size, Array64 uses S64.

template <class T, typename S> class ArrayBase
{
private:
    enum
    {
        MinBytes        = 256,                                          // Minimum number of bytes to allocate when the first element is being added.
    };

public:

    // Constructors.

    inline              ArrayBase       (void);                             // Create an empty ArrayBase. Memory is allocated lazily.
    inline explicit     ArrayBase       (const T& item);                    // Create an ArrayBase containing one element.
    inline              ArrayBase       (const T* ptr, S size);             // Copy contents from the given memory location. If NULL, the elements are left uninitialized.
    inline              ArrayBase       (const ArrayBase<T,S>& other);      // Copy constructor.
    inline              ~ArrayBase      (void);

    // ArrayBase-wide getters.

    inline S            getSize     (void) const;                       // Returns the number of elements contained in the ArrayBase.
    inline S            getCapacity (void) const;                       // Returns the number of elements currently allocated. Can be larger than getSize().
    inline const T*     getPtr      (S idx = 0) const;                  // Returns a pointer to the specified element.
    inline T*           getPtr      (S idx = 0);
    inline S            getStride   (void) const;                       // Returns the size of one element in bytes.
    inline S            getNumBytes (void) const;                       // Returns the size of the entire ArrayBase in bytes.

    // Element access.

    inline const T&         get         (S idx) const;                              // Returns a reference to the specified element.
    inline T&               get         (S idx);
    inline T                set         (S idx, const T& item);                     // Overwrites the specified element and returns the old value.
    inline const T&     getFirst    (void) const;                       // Returns a reference to the first element.
    inline T&           getFirst    (void);
    inline const T&     getLast     (void) const;                       // Returns a reference to the last element.
    inline T&           getLast     (void);
    inline void             getRange    (S start, S end, T* ptr) const;             // Copies a range of elements (start..end-1) into the given memory location.
    inline ArrayBase<T,S>   getRange    (S start, S end) const;                     // Copies a range of elements (start..end-1) into a newly allocated ArrayBase.
    inline void             setRange    (S start, S end, const T* ptr);             // Overwrites a range of elements (start..end-1) from the given memory location.
    inline void             setRange    (S start, const ArrayBase<T,S>& other);	    // Overwrites a range of elements from the given ArrayBase.

    // ArrayBase-wide operations that may shrink the allocation.

    inline void         reset       (S size = 0);                       // Discards old contents, and resets size & capacity exactly to the given value.
    inline void         setCapacity (S numElements);                    // Resizes the allocation for exactly the given number of elements. Does not modify contents.
    inline void         compact     (void);                             // Shrinks the allocation to the match the current size. Does not modify contents.
    inline void         set         (const T* ptr, S size);             // Discards old contents, and re-initializes the ArrayBase from the given memory location.
    inline void         set         (const ArrayBase<T,S>& other);      // Discards old contents, and re-initializes the ArrayBase by cloning the given ArrayBase.

    // ArrayBase-wide operations that can only grow the allocation.

    inline void         clear       (void);                             // Sets the size to zero. Does not shrink the allocation.
    inline void         resize      (S size);                           // Sets the size to the given value. Allocates more space if necessary.
    inline void         reserve     (S numElements);                    // Grows the allocation to contain at least the given number of elements. Does not modify contents.

    // Element addition. Allocates more space if necessary.

    inline T&           add         (void);                             // Adds one element and returns a reference to it.
    inline T&           add         (const T& item);                    // Adds one element and initializes it to the given value.
    inline T*           add         (const T* ptr, S size);               // Appends a number of elements from the given memory location. If NULL, the elements are left uninitialized.
    inline T*           add         (const ArrayBase<T,S>& other);        // Appends elements from the given ArrayBase.
    inline T&           insert      (S idx);                              // Inserts a new element at the given index and returns a reference to it. Shifts the following elements up.
    inline T&           insert      (S idx, const T& item);               // Inserts a new element at the given index and initializes it to the given value. Shifts the following elements up.
    inline T*           insert      (S idx, const T* ptr, S size);        // Inserts a number of elements from the given memory location. If NULL, the elements are left uninitialized.
    inline T*           insert      (S idx, const ArrayBase<T,S>& other); // Inserts elements from the given ArrayBase.

    // Element removal. Does not shrink the allocation.

    inline T            remove      (S idx);                                       // Removes the given element and returns its value. Shifts the following elements down.
    inline void         remove      (S start, S end);                              // Removes a range of elements (start..end-1). Shifts the following elements down.
    inline T&           removeLast  (void);                             // Removes the last element and returns a reference to its value.
    inline T            removeSwap  (S idx);                                       // Removes the given element and returns its value. Swaps in the last element to fill the vacant slot.
    inline void         removeSwap  (S start, S end);                              // Removes a range of elements (start..end-1). Swaps in the N last element to fill the vacant slots.
    inline T*           replace     (S start, S end, S size);                      // remove(start, end), insert(start, NULL, size)
    inline T*           replace     (S start, S end, const T* ptr, S size);        // remove(start, end), insert(start, ptr, size)
    inline T*           replace     (S start, S end, const ArrayBase<T,S>& other); // remove(start, end), insert(start, other)

    // Element search.

    inline S            indexOf     (const T& item, S fromIdx = 0) const;  // Finds the first element that equals the given value, or -1 if not found.
    inline S            lastIndexOf (const T& item) const;                 // Finds the last element that equals the given value, or -1 if not found.
    inline S            lastIndexOf (const T& item, S fromIdx) const;
    inline bool         contains    (const T& item) const;                 // Checks whether the ArrayBase contains an element that equals the given value.
    inline bool         removeItem  (const T& item);                    // Finds the first element that equals the given value and removes it.

    // Operators.

    inline const T&         operator[]  (S idx) const;
    inline T&               operator[]  (S idx);
    inline ArrayBase<T,S>&  operator=   (const ArrayBase<T,S>& other);
    inline bool             operator==  (const ArrayBase<T,S>& other) const;
    inline bool             operator!=  (const ArrayBase<T,S>& other) const;

    // Type-specific utilities.

    static inline void  copy        (T* dst, const T* src, S size);   // Analogous to memcpy().
    static inline void  copyOverlap (T* dst, const T* src, S size);   // Analogous to memmove().

    // Internals.

private:
    inline void         init        (void);
    void                realloc     (S size);
    void                reallocRound(S size);

private:
    T*                  m_ptr;
    S                   m_size;
    S                   m_alloc;
};

//------------------------------------------------------------------------

template <class T, typename S> ArrayBase<T,S>::ArrayBase(void)
{
    init();
}

//------------------------------------------------------------------------

template <class T, typename S> ArrayBase<T,S>::ArrayBase(const T& item)
{
    init();
    add(item);
}

//------------------------------------------------------------------------

template <class T, typename S> ArrayBase<T,S>::ArrayBase(const T* ptr, S size)
{
    init();
    set(ptr, size);
}

//------------------------------------------------------------------------

template <class T, typename S> ArrayBase<T,S>::ArrayBase(const ArrayBase<T,S>& other)
{
    init();
    set(other);
}

//------------------------------------------------------------------------

template <class T, typename S> ArrayBase<T,S>::~ArrayBase(void)
{
    delete[] m_ptr;
}

//------------------------------------------------------------------------

template <class T, typename S> S ArrayBase<T,S>::getSize(void) const
{
    return m_size;
}

//------------------------------------------------------------------------

template <class T, typename S> S ArrayBase<T,S>::getCapacity(void) const
{
    return m_alloc;
}

//------------------------------------------------------------------------

template <class T, typename S> const T* ArrayBase<T,S>::getPtr(S idx) const
{
    FW_ASSERT(idx >= 0 && idx <= m_size);
    return m_ptr + idx;
}

//------------------------------------------------------------------------

template <class T, typename S> T* ArrayBase<T,S>::getPtr(S idx)
{
    FW_ASSERT(idx >= 0 && idx <= m_size);
    return m_ptr + idx;
}

//------------------------------------------------------------------------

template <class T, typename S> S ArrayBase<T,S>::getStride(void) const
{
    return sizeof(T);
}

//------------------------------------------------------------------------

template <class T, typename S> S ArrayBase<T,S>::getNumBytes(void) const
{
    return getSize() * getStride();
}

//------------------------------------------------------------------------

template <class T, typename S> const T& ArrayBase<T,S>::get(S idx) const
{
    FW_ASSERT(idx >= 0 && idx < m_size);
    return m_ptr[idx];
}

//------------------------------------------------------------------------

template <class T, typename S> T& ArrayBase<T,S>::get(S idx)
{
    FW_ASSERT(idx >= 0 && idx < m_size);
    return m_ptr[idx];
}

//------------------------------------------------------------------------

template <class T, typename S> T ArrayBase<T,S>::set(S idx, const T& item)
{
    T& slot = get(idx);
    T old = slot;
    slot = item;
    return old;
}

//------------------------------------------------------------------------

template <class T, typename S> const T& ArrayBase<T,S>::getFirst(void) const
{
    return get(0);
}

//------------------------------------------------------------------------

template <class T, typename S> T& ArrayBase<T,S>::getFirst(void)
{
    return get(0);
}

//------------------------------------------------------------------------

template <class T, typename S> const T& ArrayBase<T,S>::getLast(void) const
{
    return get(getSize() - 1);
}

//------------------------------------------------------------------------

template <class T, typename S> T& ArrayBase<T,S>::getLast(void)
{
    return get(getSize() - 1);
}

//------------------------------------------------------------------------

template <class T, typename S> void ArrayBase<T,S>::getRange(S start, S end, T* ptr) const
{
    FW_ASSERT(end <= m_size);
    copy(ptr, getPtr(start), end - start);
}

//------------------------------------------------------------------------

template <class T, typename S> ArrayBase<T,S> ArrayBase<T,S>::getRange(S start, S end) const
{
    FW_ASSERT(end <= m_size);
    return ArrayBase<T,S>(getPtr(start), end - start);
}

//------------------------------------------------------------------------

template <class T, typename S> void ArrayBase<T,S>::setRange(S start, S end, const T* ptr)
{
    FW_ASSERT(end <= m_size);
    copy(getPtr(start), ptr, end - start);
}

//------------------------------------------------------------------------

template <class T, typename S> void ArrayBase<T,S>::setRange(S start, const ArrayBase<T,S>& other)
{
    setRange(start, start + other.getSize(), other.getPtr());
}

//------------------------------------------------------------------------

template <class T, typename S> void ArrayBase<T,S>::reset(S size)
{
    clear();
    setCapacity(size);
    m_size = size;
}

//------------------------------------------------------------------------

template <class T, typename S> void ArrayBase<T,S>::setCapacity(S numElements)
{
    S c = max(numElements, m_size);
    if (m_alloc != c)
        realloc(c);
}

//------------------------------------------------------------------------

template <class T, typename S> void ArrayBase<T,S>::compact(void)
{
    setCapacity(0);
}

//------------------------------------------------------------------------

template <class T, typename S> void ArrayBase<T,S>::set(const T* ptr, S size)
{
    reset(size);
    if (ptr)
        copy(getPtr(), ptr, size);
}

//------------------------------------------------------------------------

template <class T, typename S> void ArrayBase<T,S>::set(const ArrayBase<T,S>& other)
{
    if (&other != this)
        set(other.getPtr(), other.getSize());
}

//------------------------------------------------------------------------

template <class T, typename S> void ArrayBase<T,S>::clear(void)
{
    m_size = 0;
}

//------------------------------------------------------------------------

template <class T, typename S> void ArrayBase<T,S>::resize(S size)
{
    FW_ASSERT(size >= 0);
    if (size > m_alloc)
        reallocRound(size);
    m_size = size;
}

//------------------------------------------------------------------------

template <class T, typename S> void ArrayBase<T,S>::reserve(S numElements)
{
    if (numElements > m_alloc)
        realloc(numElements);
}

//------------------------------------------------------------------------

template <class T, typename S> T& ArrayBase<T,S>::add(void)
{
    return *add(NULL, 1);
}

//------------------------------------------------------------------------

template <class T, typename S> T& ArrayBase<T,S>::add(const T& item)
{
    T* slot = add(NULL, 1);
    *slot = item;
    return *slot;
}

//------------------------------------------------------------------------

template <class T, typename S> T* ArrayBase<T,S>::add(const T* ptr, S size)
{
    S oldSize = getSize();
    resize(oldSize + size);
    T* slot = getPtr(oldSize);
    if (ptr)
        copy(slot, ptr, size);
    return slot;
}

//------------------------------------------------------------------------

template <class T, typename S> T* ArrayBase<T,S>::add(const ArrayBase<T,S>& other)
{
    return replace(getSize(), getSize(), other);
}

//------------------------------------------------------------------------

template <class T, typename S> T& ArrayBase<T,S>::insert(S idx)
{
    return *replace(idx, idx, 1);
}

//------------------------------------------------------------------------

template <class T, typename S> T& ArrayBase<T,S>::insert(S idx, const T& item)
{
    T* slot = replace(idx, idx, 1);
    *slot = item;
    return *slot;
}

//------------------------------------------------------------------------

template <class T, typename S> T* ArrayBase<T,S>::insert(S idx, const T* ptr, S size)
{
    return replace(idx, idx, ptr, size);
}

//------------------------------------------------------------------------

template <class T, typename S> T* ArrayBase<T,S>::insert(S idx, const ArrayBase<T,S>& other)
{
    return replace(idx, idx, other);
}

//------------------------------------------------------------------------

template <class T, typename S> T ArrayBase<T,S>::remove(S idx)
{
    T old = get(idx);
    replace(idx, idx + 1, 0);
    return old;
}

//------------------------------------------------------------------------

template <class T, typename S> void ArrayBase<T,S>::remove(S start, S end)
{
    replace(start, end, 0);
}

//------------------------------------------------------------------------

template <class T, typename S> T& ArrayBase<T,S>::removeLast(void)
{
    FW_ASSERT(m_size > 0);
    m_size--;
    return m_ptr[m_size];
}

//------------------------------------------------------------------------

template <class T, typename S> T ArrayBase<T,S>::removeSwap(S idx)
{
    FW_ASSERT(idx >= 0 && idx < m_size);

    T old = get(idx);
    m_size--;
    if (idx < m_size)
        m_ptr[idx] = m_ptr[m_size];
    return old;
}

//------------------------------------------------------------------------

template <class T, typename S> void ArrayBase<T,S>::removeSwap(S start, S end)
{
    FW_ASSERT(start >= 0);
    FW_ASSERT(start <= end);
    FW_ASSERT(end <= m_size);

    S oldSize = m_size;
    m_size += start - end;

    S copyStart = max(m_size, end);
    copy(m_ptr + start, m_ptr + copyStart, oldSize - copyStart);
}

//------------------------------------------------------------------------

template <class T, typename S> T* ArrayBase<T,S>::replace(S start, S end, S size)
{
    FW_ASSERT(start >= 0);
    FW_ASSERT(start <= end);
    FW_ASSERT(end <= m_size);
    FW_ASSERT(size >= 0);

    S tailSize = m_size - end;
    S newEnd = start + size;
    resize(m_size + newEnd - end);

    copyOverlap(m_ptr + newEnd, m_ptr + end, tailSize);
    return m_ptr + start;
}

//------------------------------------------------------------------------

template <class T, typename S> T* ArrayBase<T,S>::replace(S start, S end, const T* ptr, S size)
{
    T* slot = replace(start, end, size);
    if (ptr)
        copy(slot, ptr, size);
    return slot;
}

//------------------------------------------------------------------------

template <class T, typename S> T* ArrayBase<T,S>::replace(S start, S end, const ArrayBase<T,S>& other)
{
    ArrayBase<T,S> tmp;
    const T* ptr = other.getPtr();
    if (&other == this)
    {
        tmp = other;
        ptr = tmp.getPtr();
    }
    return replace(start, end, ptr, other.getSize());
}

//------------------------------------------------------------------------

template <class T, typename S> S ArrayBase<T,S>::indexOf(const T& item, S fromIdx) const
{
    for (S i = max(fromIdx, 0); i < getSize(); i++)
        if (get(i) == item)
            return i;
    return -1;
}

//------------------------------------------------------------------------

template <class T, typename S> S ArrayBase<T,S>::lastIndexOf(const T& item) const
{
    return lastIndexOf(item, getSize() - 1);
}

//------------------------------------------------------------------------

template <class T, typename S> S ArrayBase<T,S>::lastIndexOf(const T& item, S fromIdx) const
{
    for (S i = min(fromIdx, getSize() - 1); i >= 0; i--)
        if (get(i) == item)
            return i;
    return -1;
}

//------------------------------------------------------------------------

template <class T, typename S> bool ArrayBase<T,S>::contains(const T& item) const
{
    return (indexOf(item) != -1);
}

//------------------------------------------------------------------------

template <class T, typename S> bool ArrayBase<T,S>::removeItem(const T& item)
{
    S idx = indexOf(item);
    if (idx == -1)
        return false;
    remove(idx);
    return true;
}

//------------------------------------------------------------------------

template <class T, typename S> const T& ArrayBase<T,S>::operator[](S idx) const
{
    return get(idx);
}

//------------------------------------------------------------------------

template <class T, typename S> T& ArrayBase<T,S>::operator[](S idx)
{
    return get(idx);
}

//------------------------------------------------------------------------

template <class T, typename S> ArrayBase<T,S>& ArrayBase<T,S>::operator=(const ArrayBase<T,S>& other)
{
    set(other);
    return *this;
}

//------------------------------------------------------------------------

template <class T, typename S> bool ArrayBase<T,S>::operator==(const ArrayBase<T,S>& other) const
{
    if (getSize() != other.getSize())
        return false;

    for (S i = 0; i < getSize(); i++)
        if (get(i) != other[i])
            return false;
    return true;
}

//------------------------------------------------------------------------

template <class T, typename S> bool ArrayBase<T,S>::operator!=(const ArrayBase<T,S>& other) const
{
    return (!operator==(other));
}

//------------------------------------------------------------------------

template <class T, typename S> void ArrayBase<T,S>::copy(T* dst, const T* src, S size)
{
    FW_ASSERT(size >= 0);
    if (!size)
        return;

    FW_ASSERT(dst && src);
    for (S i = 0; i < size; i++)
        dst[i] = src[i];
}

//------------------------------------------------------------------------

template <class T, typename S> void ArrayBase<T,S>::copyOverlap(T* dst, const T* src, S size)
{
    FW_ASSERT(size >= 0);
    if (!size)
        return;

    FW_ASSERT(dst && src);
    if (dst < src || dst >= src + size)
        for (S i = 0; i < size; i++)
            dst[i] = src[i];
    else
        for (S i = size - 1; i >= 0; i--)
            dst[i] = src[i];
}

//------------------------------------------------------------------------

template <class T, typename S> void ArrayBase<T,S>::init(void)
{
    m_ptr = NULL;
    m_size = 0;
    m_alloc = 0;
}

//------------------------------------------------------------------------

template <class T, typename S> void ArrayBase<T,S>::realloc(S size)
{
    FW_ASSERT(size >= 0);

    T* newPtr = NULL;
    if (size)
    {
        newPtr = new T[size];
        copy(newPtr, m_ptr, min(size, m_size));
    }

    delete[] m_ptr;
    m_ptr = newPtr;
    m_alloc = size;
}

//------------------------------------------------------------------------

template <class T, typename S> void ArrayBase<T,S>::reallocRound(S size)
{
    FW_ASSERT(size >= 0);
    S rounded = max((S)(MinBytes / sizeof(T)), S(1));
    while (size > rounded)
        rounded <<= 1;
    realloc(rounded);
}

//------------------------------------------------------------------------

inline void ArrayBase<S8,S32>::copy(S8* dst, const S8* src, int size)           { memcpy(dst, src, size * sizeof(S8)); }
inline void ArrayBase<U8,S32>::copy(U8* dst, const U8* src, int size)           { memcpy(dst, src, size * sizeof(U8)); }
inline void ArrayBase<S16,S32>::copy(S16* dst, const S16* src, int size)        { memcpy(dst, src, size * sizeof(S16)); }
inline void ArrayBase<U16,S32>::copy(U16* dst, const U16* src, int size)        { memcpy(dst, src, size * sizeof(U16)); }
inline void ArrayBase<S32,S32>::copy(S32* dst, const S32* src, int size)        { memcpy(dst, src, size * sizeof(S32)); }
inline void ArrayBase<U32,S32>::copy(U32* dst, const U32* src, int size)        { memcpy(dst, src, size * sizeof(U32)); }
inline void ArrayBase<F32,S32>::copy(F32* dst, const F32* src, int size)        { memcpy(dst, src, size * sizeof(F32)); }
inline void ArrayBase<S64,S32>::copy(S64* dst, const S64* src, int size)        { memcpy(dst, src, size * sizeof(S64)); }
inline void ArrayBase<U64,S32>::copy(U64* dst, const U64* src, int size)        { memcpy(dst, src, size * sizeof(U64)); }
inline void ArrayBase<F64,S32>::copy(F64* dst, const F64* src, int size)        { memcpy(dst, src, size * sizeof(F64)); }

inline void ArrayBase<Vec2i,S32>::copy(Vec2i* dst, const Vec2i* src, int size)  { memcpy(dst, src, size * sizeof(Vec2i)); }
inline void ArrayBase<Vec2f,S32>::copy(Vec2f* dst, const Vec2f* src, int size)  { memcpy(dst, src, size * sizeof(Vec2f)); }
inline void ArrayBase<Vec3i,S32>::copy(Vec3i* dst, const Vec3i* src, int size)  { memcpy(dst, src, size * sizeof(Vec3i)); }
inline void ArrayBase<Vec3f,S32>::copy(Vec3f* dst, const Vec3f* src, int size)  { memcpy(dst, src, size * sizeof(Vec3f)); }
inline void ArrayBase<Vec4i,S32>::copy(Vec4i* dst, const Vec4i* src, int size)  { memcpy(dst, src, size * sizeof(Vec4i)); }
inline void ArrayBase<Vec4f,S32>::copy(Vec4f* dst, const Vec4f* src, int size)  { memcpy(dst, src, size * sizeof(Vec4f)); }

inline void ArrayBase<Mat2f,S32>::copy(Mat2f* dst, const Mat2f* src, int size)  { memcpy(dst, src, size * sizeof(Mat2f)); }
inline void ArrayBase<Mat3f,S32>::copy(Mat3f* dst, const Mat3f* src, int size)  { memcpy(dst, src, size * sizeof(Mat3f)); }
inline void ArrayBase<Mat4f,S32>::copy(Mat4f* dst, const Mat4f* src, int size)  { memcpy(dst, src, size * sizeof(Mat4f)); }

//------------------------------------------------------------------------

inline void ArrayBase<S8,S64>::copy(S8* dst, const S8* src, S64 size)           { memcpy(dst, src, (size_t)size * sizeof(S8)); }
inline void ArrayBase<U8,S64>::copy(U8* dst, const U8* src, S64 size)           { memcpy(dst, src, (size_t)size * sizeof(U8)); }
inline void ArrayBase<S16,S64>::copy(S16* dst, const S16* src, S64 size)        { memcpy(dst, src, (size_t)size * sizeof(S16)); }
inline void ArrayBase<U16,S64>::copy(U16* dst, const U16* src, S64 size)        { memcpy(dst, src, (size_t)size * sizeof(U16)); }
inline void ArrayBase<S32,S64>::copy(S32* dst, const S32* src, S64 size)        { memcpy(dst, src, (size_t)size * sizeof(S32)); }
inline void ArrayBase<U32,S64>::copy(U32* dst, const U32* src, S64 size)        { memcpy(dst, src, (size_t)size * sizeof(U32)); }
inline void ArrayBase<F32,S64>::copy(F32* dst, const F32* src, S64 size)        { memcpy(dst, src, (size_t)size * sizeof(F32)); }
inline void ArrayBase<S64,S64>::copy(S64* dst, const S64* src, S64 size)        { memcpy(dst, src, (size_t)size * sizeof(S64)); }
inline void ArrayBase<U64,S64>::copy(U64* dst, const U64* src, S64 size)        { memcpy(dst, src, (size_t)size * sizeof(U64)); }
inline void ArrayBase<F64,S64>::copy(F64* dst, const F64* src, S64 size)        { memcpy(dst, src, (size_t)size * sizeof(F64)); }

inline void ArrayBase<Vec2i,S64>::copy(Vec2i* dst, const Vec2i* src, S64 size)  { memcpy(dst, src, (size_t)size * sizeof(Vec2i)); }
inline void ArrayBase<Vec2f,S64>::copy(Vec2f* dst, const Vec2f* src, S64 size)  { memcpy(dst, src, (size_t)size * sizeof(Vec2f)); }
inline void ArrayBase<Vec3i,S64>::copy(Vec3i* dst, const Vec3i* src, S64 size)  { memcpy(dst, src, (size_t)size * sizeof(Vec3i)); }
inline void ArrayBase<Vec3f,S64>::copy(Vec3f* dst, const Vec3f* src, S64 size)  { memcpy(dst, src, (size_t)size * sizeof(Vec3f)); }
inline void ArrayBase<Vec4i,S64>::copy(Vec4i* dst, const Vec4i* src, S64 size)  { memcpy(dst, src, (size_t)size * sizeof(Vec4i)); }
inline void ArrayBase<Vec4f,S64>::copy(Vec4f* dst, const Vec4f* src, S64 size)  { memcpy(dst, src, (size_t)size * sizeof(Vec4f)); }

inline void ArrayBase<Mat2f,S64>::copy(Mat2f* dst, const Mat2f* src, S64 size)  { memcpy(dst, src, (size_t)size * sizeof(Mat2f)); }
inline void ArrayBase<Mat3f,S64>::copy(Mat3f* dst, const Mat3f* src, S64 size)  { memcpy(dst, src, (size_t)size * sizeof(Mat3f)); }
inline void ArrayBase<Mat4f,S64>::copy(Mat4f* dst, const Mat4f* src, S64 size)  { memcpy(dst, src, (size_t)size * sizeof(Mat4f)); }

//------------------------------------------------------------------------

template<class T> class Array : public ArrayBase<T,S32>
{
public:
	inline              Array       (void) : ArrayBase<T,S32>() { };
	inline explicit     Array       (const T& item) : ArrayBase<T,S32>( item ) { };   
	inline              Array       (const T* ptr, S32 size) : ArrayBase<T,S32>( ptr, size ) { };
	inline              Array       (const Array<T>& other) : ArrayBase<T,S32>( other ) { };
};

template<class T> class Array64 : public ArrayBase<T,S64>
{
public:
	inline              Array64     (void) : ArrayBase<T,S64>() { };
	inline explicit     Array64     (const T& item) : ArrayBase<T,S64>( item ) { };   
	inline              Array64     (const T* ptr, S64 size) : ArrayBase<T,S64>( ptr, size ) { };
	inline              Array64     (const Array64<T>& other) : ArrayBase<T,S64>( other ) { };
};


}
