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
#include "base/String.hpp"

namespace FW
{

//------------------------------------------------------------------------

template <class T> class Set
{
private:
    enum
    {
        BlockSize   = 8,
        MinBytes    = 32,
        MaxUsagePct = 60,
        ThrUsagePct = MaxUsagePct * 3 / 4
    };

    enum HashValue
    {
        Empty       = -1,
        Removed     = -2
    };

public:
                        Set         (void)                          { init(); }
                        Set         (const Set<T>& other)           { init(); set(other); }
                        ~Set        (void)                          { reset(); }

    int                 getSize     (void) const                    { return m_numItems; }
    bool                contains    (const T& value) const          { return (findSlot(value) != -1); }
    const T*            search      (const T& value) const          { int slot = findSlot(value); return (slot == -1) ? NULL : &m_values[slot]; }
    T*                  search      (const T& value)                { int slot = findSlot(value); return (slot == -1) ? NULL : &m_values[slot]; }
    const T&            get         (const T& value) const          { int slot = findSlot(value); FW_ASSERT(slot != -1); return m_values[slot]; }
    T&                  get         (const T& value)                { int slot = findSlot(value); FW_ASSERT(slot != -1); return m_values[slot]; }

    void                clear       (void)                          { m_numItems = 0; m_numNonEmpty = 0; memset(m_hashes, Empty, m_capacity * sizeof(S32)); }
    void                reset       (void)                          { delete[] m_hashes; delete[] m_values; init(); }
    void                setCapacity (int numItems);
    void                compact     (void)                          { setCapacity(m_numItems); }
    void                set         (const Set<T>& other);

    T&                  add         (const T& value)                { T& slot = addNoAssign(value); slot = value; return slot; }
    T&                  addNoAssign (const T& value);
    T&                  remove      (const T& value);
    T                   replace     (const T& value);

    int                 findSlot    (const T& value) const          { return findSlot(value, hash<T>(value) >> 1, false); }
    int                 firstSlot   (void) const                    { return nextSlot(-1); }
    int                 nextSlot    (int slot) const;
    const T&            getSlot     (int slot) const                { FW_ASSERT(m_hashes[slot] >= 0); return m_values[slot]; }
    T&                  getSlot     (int slot)                      { FW_ASSERT(m_hashes[slot] >= 0); return m_values[slot]; }

    Set<T>&             operator=   (const Set<T>& other)           { set(other); return *this; }
    const T&            operator[]  (const T& value) const          { return get(value); }
    T&                  operator[]  (const T& value)                { return get(value); }

private:
    void                init        (void)                          { m_capacity = 0; m_numItems = 0; m_numNonEmpty = 0; m_hashes = NULL; m_values = NULL; }
    int                 findSlot    (const T& value, S32 hashValue, bool needEmpty) const;
    void                rehash      (int capacity);

private:
    S32                 m_capacity;
    S32                 m_numItems;
    S32                 m_numNonEmpty;
    S32*                m_hashes;
    T*                  m_values;
};

//------------------------------------------------------------------------

template <class K, class V> struct HashEntry
{
    K                   key;
    V                   value;
};

//------------------------------------------------------------------------

template <class K, class V> class Hash
{
public:
    typedef HashEntry<K, V> Entry;

public:
                        Hash        (void)                          {}
                        Hash        (const Hash<K, V>& other)       { set(other); }
                        ~Hash       (void)                          {}

    const Set<Entry>&   getEntries  (void) const                    { return m_entries; }
    Set<Entry>&         getEntries  (void)                          { return m_entries; }
    int                 getSize     (void) const                    { return m_entries.getSize(); }
    bool                contains    (const K& key) const            { return m_entries.contains(keyEntry(key)); }
    const Entry*        searchEntry (const K& key) const            { return m_entries.search(keyEntry(key)); }
    Entry*              searchEntry (const K& key)                  { return m_entries.search(keyEntry(key)); }
    const K*            searchKey   (const K& key) const            { const Entry* e = searchEntry(key); return (e) ? &e->key : NULL; }
    K*                  searchKey   (const K& key)                  { Entry* e = searchEntry(key); return (e) ? &e->key : NULL; }
    const V*            search      (const K& key) const            { const Entry* e = searchEntry(key); return (e) ? &e->value : NULL; }
    V*                  search      (const K& key)                  { Entry* e = searchEntry(key); return (e) ? &e->value : NULL; }
    const Entry&        getEntry    (const K& key) const            { return m_entries.get(keyEntry(key)); }
    Entry&              getEntry    (const K& key)                  { return m_entries.get(keyEntry(key)); }
    const K&            getKey      (const K& key) const            { return getEntry(key).key; }
    K&                  getKey      (const K& key)                  { return getEntry(key).key; }
    const V&            get         (const K& key) const            { return getEntry(key).value; }
    V&                  get         (const K& key)                  { return getEntry(key).value; }

    void                clear       (void)                          { m_entries.clear(); }
    void                reset       (void)                          { m_entries.reset(); }
    void                setCapacity (int numItems)                  { m_entries.setCapacity(numItems); }
    void                compact     (void)                          { m_entries.compact(); }
    void                set         (const Hash<K, V>& other)       { m_entries.set(other.m_entries); }

    V&                  add         (const K& key, const V& value)  { Entry& slot = m_entries.addNoAssign(keyEntry(key)); slot.key = key; slot.value = value; return slot.value; }
    V&                  add         (const K& key)                  { Entry& slot = m_entries.addNoAssign(keyEntry(key)); slot.key = key; return slot.value; }
    V&                  remove      (const K& key)                  { return m_entries.remove(keyEntry(key)).value; }
    V                   replace     (const K& key, const V& value)  { Entry e; e.key = key; e.value = value; return m_entries.replace(e).value; }

    int                 findSlot    (const K& key) const            { return m_entries.findSlot(keyEntry(key)); }
    int                 firstSlot   (void) const                    { return m_entries.firstSlot(); }
    int                 nextSlot    (int slot) const                { return m_entries.nextSlot(slot); }
    const Entry&        getSlot     (int slot) const                { return m_entries.getSlot(slot); }
    Entry&              getSlot     (int slot)                      { return m_entries.getSlot(slot); }

    Hash<K, V>&         operator=   (const Hash<K, V>& other)       { set(other); return *this; }
    const V&            operator[]  (const K& key) const            { return get(key); }
    V&                  operator[]  (const K& key)                  { return get(key); }

private:
    static const Entry& keyEntry    (const K& key)                  { return *(Entry*)&key; }

private:
    Set<Entry>          m_entries;
};

//------------------------------------------------------------------------
// Helpers for equals() and hash().
//------------------------------------------------------------------------

#define FW_HASH_MAGIC   (0x9e3779b9u)

// By Bob Jenkins, 1996. bob_jenkins@burtleburtle.net.
#define FW_JENKINS_MIX(a, b, c)   \
    a -= b; a -= c; a ^= (c>>13); \
    b -= c; b -= a; b ^= (a<<8);  \
    c -= a; c -= b; c ^= (b>>13); \
    a -= b; a -= c; a ^= (c>>12); \
    b -= c; b -= a; b ^= (a<<16); \
    c -= a; c -= b; c ^= (b>>5);  \
    a -= b; a -= c; a ^= (c>>3);  \
    b -= c; b -= a; b ^= (a<<10); \
    c -= a; c -= b; c ^= (b>>15);

inline U32  hashBits        (U32 a, U32 b = FW_HASH_MAGIC, U32 c = 0)                   { c += FW_HASH_MAGIC; FW_JENKINS_MIX(a, b, c); return c; }
inline U32  hashBits        (U32 a, U32 b, U32 c, U32 d, U32 e = 0, U32 f = 0)          { c += FW_HASH_MAGIC; FW_JENKINS_MIX(a, b, c); a += d; b += e; c += f; FW_JENKINS_MIX(a, b, c); return c; }

inline bool equalsBuffer    (const void* ptrA, const void* ptrB, int size)              { return (memcmp(ptrA, ptrB, size) == 0); }
inline bool equalsBuffer    (const void* ptrA, int sizeA, const void* ptrB, int sizeB)  { return (sizeA == sizeB && memcmp(ptrA, ptrB, sizeA) == 0); }
U32         hashBuffer      (const void* ptr, int size);
U32         hashBufferAlign (const void* ptr, int size);

//------------------------------------------------------------------------
// Base templates.
//------------------------------------------------------------------------

template <class T>  inline bool equalsArray     (const T* ptrA, int sizeA, const T* ptrB, int sizeB);
template <class T>  inline U32  hashArray       (const T* ptr, int size);

template <class T>  inline bool equals          (const T& a, const T& b)                { return equalsBuffer(&a, &b, sizeof(T)); }
template <class T>  inline U32  hash            (const T& value)                        { return hashBuffer(&value, sizeof(T)); }

//------------------------------------------------------------------------
// Specializations for primitive types.
//------------------------------------------------------------------------

template <> inline bool equalsArray<S8> (const S8* ptrA, int sizeA, const S8* ptrB, int sizeB)      { return equalsBuffer(ptrA, sizeA * (int)sizeof(S8), ptrB, sizeB * (int)sizeof(S8)); }
template <> inline bool equalsArray<U8> (const U8* ptrA, int sizeA, const U8* ptrB, int sizeB)      { return equalsBuffer(ptrA, sizeA * (int)sizeof(U8), ptrB, sizeB * (int)sizeof(U8)); }
template <> inline bool equalsArray<S16>(const S16* ptrA, int sizeA, const S16* ptrB, int sizeB)    { return equalsBuffer(ptrA, sizeA * (int)sizeof(S16), ptrB, sizeB * (int)sizeof(S16)); }
template <> inline bool equalsArray<U16>(const U16* ptrA, int sizeA, const U16* ptrB, int sizeB)    { return equalsBuffer(ptrA, sizeA * (int)sizeof(U16), ptrB, sizeB * (int)sizeof(U16)); }
template <> inline bool equalsArray<S32>(const S32* ptrA, int sizeA, const S32* ptrB, int sizeB)    { return equalsBuffer(ptrA, sizeA * (int)sizeof(S32), ptrB, sizeB * (int)sizeof(S32)); }
template <> inline bool equalsArray<U32>(const U32* ptrA, int sizeA, const U32* ptrB, int sizeB)    { return equalsBuffer(ptrA, sizeA * (int)sizeof(U32), ptrB, sizeB * (int)sizeof(U32)); }
template <> inline bool equalsArray<F32>(const F32* ptrA, int sizeA, const F32* ptrB, int sizeB)    { return equalsBuffer(ptrA, sizeA * (int)sizeof(F32), ptrB, sizeB * (int)sizeof(F32)); }
template <> inline bool equalsArray<S64>(const S64* ptrA, int sizeA, const S64* ptrB, int sizeB)    { return equalsBuffer(ptrA, sizeA * (int)sizeof(S64), ptrB, sizeB * (int)sizeof(S64)); }
template <> inline bool equalsArray<U64>(const U64* ptrA, int sizeA, const U64* ptrB, int sizeB)    { return equalsBuffer(ptrA, sizeA * (int)sizeof(U64), ptrB, sizeB * (int)sizeof(U64)); }
template <> inline bool equalsArray<F64>(const F64* ptrA, int sizeA, const F64* ptrB, int sizeB)    { return equalsBuffer(ptrA, sizeA * (int)sizeof(F64), ptrB, sizeB * (int)sizeof(F64)); }

template <> inline U32  hashArray<S8>   (const S8* ptr, int size)           { return hashBuffer(ptr, size * (int)sizeof(S8)); }
template <> inline U32  hashArray<U8>   (const U8* ptr, int size)           { return hashBuffer(ptr, size * (int)sizeof(U8)); }
template <> inline U32  hashArray<S16>  (const S16* ptr, int size)          { return hashBuffer(ptr, size * (int)sizeof(S16)); }
template <> inline U32  hashArray<U16>  (const U16* ptr, int size)          { return hashBuffer(ptr, size * (int)sizeof(U16)); }
template <> inline U32  hashArray<S32>  (const S32* ptr, int size)          { return hashBuffer(ptr, size * (int)sizeof(S32)); }
template <> inline U32  hashArray<U32>  (const U32* ptr, int size)          { return hashBuffer(ptr, size * (int)sizeof(U32)); }
template <> inline U32  hashArray<F32>  (const F32* ptr, int size)          { return hashBuffer(ptr, size * (int)sizeof(F32)); }
template <> inline U32  hashArray<S64>  (const S64* ptr, int size)          { return hashBuffer(ptr, size * (int)sizeof(S64)); }
template <> inline U32  hashArray<U64>  (const U64* ptr, int size)          { return hashBuffer(ptr, size * (int)sizeof(U64)); }
template <> inline U32  hashArray<F64>  (const F64* ptr, int size)          { return hashBuffer(ptr, size * (int)sizeof(F64)); }

template <> inline bool equals<S8>      (const S8& a, const S8& b)          { return (a == b); }
template <> inline bool equals<U8>      (const U8& a, const U8& b)          { return (a == b); }
template <> inline bool equals<S16>     (const S16& a, const S16& b)        { return (a == b); }
template <> inline bool equals<U16>     (const U16& a, const U16& b)        { return (a == b); }
template <> inline bool equals<S32>     (const S32& a, const S32& b)        { return (a == b); }
template <> inline bool equals<U32>     (const U32& a, const U32& b)        { return (a == b); }
template <> inline bool equals<F32>     (const F32& a, const F32& b)        { return (floatToBits(a) == floatToBits(b)); }
template <> inline bool equals<S64>     (const S64& a, const S64& b)        { return (a == b); }
template <> inline bool equals<U64>     (const U64& a, const U64& b)        { return (a == b); }
template <> inline bool equals<F64>     (const F64& a, const F64& b)        { return (doubleToBits(a) == doubleToBits(b)); }

template <> inline U32  hash<S8>        (const S8& value)                   { return hashBits(value); }
template <> inline U32  hash<U8>        (const U8& value)                   { return hashBits(value); }
template <> inline U32  hash<S16>       (const S16& value)                  { return hashBits(value); }
template <> inline U32  hash<U16>       (const U16& value)                  { return hashBits(value); }
template <> inline U32  hash<S32>       (const S32& value)                  { return hashBits(value); }
template <> inline U32  hash<U32>       (const U32& value)                  { return hashBits(value); }
template <> inline U32  hash<F32>       (const F32& value)                  { return hashBits(floatToBits(value)); }
template <> inline U32  hash<S64>       (const S64& value)                  { return hashBits((U32)value, (U32)(value >> 32)); }
template <> inline U32  hash<U64>       (const U64& value)                  { return hash<S64>((S64)value); }
template <> inline U32  hash<F64>       (const F64& value)                  { return hash<U64>(doubleToBits(value)); }

//------------------------------------------------------------------------
// Specializations for compound types.
//------------------------------------------------------------------------

template <> inline bool equals<Vec2i>   (const Vec2i& a, const Vec2i& b)    { return (a == b); }
template <> inline bool equals<Vec2f>   (const Vec2f& a, const Vec2f& b)    { return (equals<F32>(a.x, b.x) && equals<F32>(a.y, b.y)); }
template <> inline bool equals<Vec3i>   (const Vec3i& a, const Vec3i& b)    { return (a == b); }
template <> inline bool equals<Vec3f>   (const Vec3f& a, const Vec3f& b)    { return (equals<F32>(a.x, b.x) && equals<F32>(a.y, b.y) && equals<F32>(a.z, b.z)); }
template <> inline bool equals<Vec4i>   (const Vec4i& a, const Vec4i& b)    { return (a == b); }
template <> inline bool equals<Vec4f>   (const Vec4f& a, const Vec4f& b)    { return (equals<F32>(a.x, b.x) && equals<F32>(a.y, b.y) && equals<F32>(a.z, b.z) && equals<F32>(a.w, b.w)); }
template <> inline bool equals<Mat2f>   (const Mat2f& a, const Mat2f& b)    { return equalsBuffer(&a, &b, sizeof(a)); }
template <> inline bool equals<Mat3f>   (const Mat3f& a, const Mat3f& b)    { return equalsBuffer(&a, &b, sizeof(a)); }
template <> inline bool equals<Mat4f>   (const Mat4f& a, const Mat4f& b)    { return equalsBuffer(&a, &b, sizeof(a)); }
template <> inline bool equals<String>  (const String& a, const String& b)  { return equalsBuffer(a.getPtr(), a.getLength(), b.getPtr(), b.getLength()); }

template <> inline U32  hash<Vec2i>     (const Vec2i& value)                { return hashBits(value.x, value.y); }
template <> inline U32  hash<Vec2f>     (const Vec2f& value)                { return hashBits(floatToBits(value.x), floatToBits(value.y)); }
template <> inline U32  hash<Vec3i>     (const Vec3i& value)                { return hashBits(value.x, value.y, value.z); }
template <> inline U32  hash<Vec3f>     (const Vec3f& value)                { return hashBits(floatToBits(value.x), floatToBits(value.y), floatToBits(value.z)); }
template <> inline U32  hash<Vec4i>     (const Vec4i& value)                { return hashBits(value.x, value.y, value.z, value.w); }
template <> inline U32  hash<Vec4f>     (const Vec4f& value)                { return hashBits(floatToBits(value.x), floatToBits(value.y), floatToBits(value.z), floatToBits(value.w)); }
template <> inline U32  hash<Mat2f>     (const Mat2f& value)                { return hashBufferAlign(&value, sizeof(value)); }
template <> inline U32  hash<Mat3f>     (const Mat3f& value)                { return hashBufferAlign(&value, sizeof(value)); }
template <> inline U32  hash<Mat4f>     (const Mat4f& value)                { return hashBufferAlign(&value, sizeof(value)); }
template <> inline U32  hash<String>    (const String& value)               { return hashBuffer(value.getPtr(), value.getLength()); }

//------------------------------------------------------------------------
// Partial specializations.
//------------------------------------------------------------------------

template <class T, class TT> inline bool equals(TT* const& a, TT* const& b) { return (a == b); }
template <class T, class TT> inline U32 hash(TT* const& value) { return hashBits((U32)(UPTR)value); }

template <class T, class K, class V> inline bool equals(const HashEntry<K, V>& a, const HashEntry<K, V>& b) { return equals<K>(a.key, b.key); }
template <class T, class K, class V> inline U32 hash (const HashEntry<K, V>& value) { return hash<K>(value.key); }

template <class T, class TT> inline bool equals(const Array<TT>& a, const Array<TT>& b) { return equalsArray<T>(a.getPtr(), a.getSize(), b.getPtr(), b.getSize()); }
template <class T, class TT> inline U32  hash(const Array<TT>& value) { return hashArray<T>(value.getPtr(), value.getSize()); }

//------------------------------------------------------------------------
// GenericHashKey.
//------------------------------------------------------------------------

struct GenericHashKey
{
    const void* ptr;
    S32         size;

    GenericHashKey(void) : ptr(NULL), size(0) {}
    GenericHashKey(const void* p, int s) : ptr(p), size(s) { FW_ASSERT(s >= 0); FW_ASSERT(p || !s); }
    template <class T> GenericHashKey(const T* p) : ptr(p), size(sizeof(T)) { FW_ASSERT(p); }
};

template <> inline bool equals<GenericHashKey>(const GenericHashKey& a, const GenericHashKey& b) { return equalsBuffer(a.ptr, a.size, b.ptr, b.size); }
template <> inline U32 hash<GenericHashKey>(const GenericHashKey& value) { return hashBuffer(value.ptr, value.size); }

//------------------------------------------------------------------------
// Implementation.
//------------------------------------------------------------------------

template <class T> void Set<T>::setCapacity(int numItems)
{
    int capacity = BlockSize;
    S64 limit = (S64)max(numItems, m_numItems, (MinBytes + (S32)sizeof(T) - 1) / (S32)sizeof(T)) * 100;
    while ((S64)capacity * MaxUsagePct < limit)
        capacity <<= 1;

    if (capacity != m_capacity)
        rehash(capacity);
}

//------------------------------------------------------------------------

template <class T> void Set<T>::set(const Set<T>& other)
{
    if (this == &other)
        return;

    clear();
    if (!other.m_numItems)
        return;

    m_capacity      = other.m_capacity;
    m_numItems      = other.m_numItems;
    m_numNonEmpty   = other.m_numNonEmpty;
    m_hashes        = new S32[m_capacity];
    m_values        = new T[m_capacity];

    memcpy(m_hashes, other.m_hashes, m_capacity * sizeof(S32));
    for (int i = 0; i < m_capacity; i++)
        m_values[i] = other.m_values[i];
}

//------------------------------------------------------------------------

template <class T> T& Set<T>::addNoAssign(const T& value)
{
    FW_ASSERT(!contains(value));

    // Empty => allocate.

    if (!m_capacity)
        setCapacity(0);

    // Exceeds MaxUsagePct => rehash.

    else if ((S64)m_numNonEmpty * 100 >= (S64)m_capacity * MaxUsagePct)
    {
        int cap = m_capacity;
        if ((S64)m_numItems * 100 >= (S64)cap * ThrUsagePct)
            cap <<= 1;
        rehash(cap);
    }

    // Find slot.

    S32 hashValue = hash<T>(value) >> 1;
    int slot = findSlot(value, hashValue, true);
    FW_ASSERT(m_hashes[slot] < 0);

    // Add item.

    m_numItems++;
    if (m_hashes[slot] == Empty)
        m_numNonEmpty++;

    m_hashes[slot] = hashValue;
    return m_values[slot];
}

//------------------------------------------------------------------------

template <class T> T& Set<T>::remove(const T& value)
{
    int slot = findSlot(value, hash<T>(value) >> 1, false);
    FW_ASSERT(m_hashes[slot] >= 0);

    m_numItems--;
    m_hashes[slot] = Removed;
    return m_values[slot];
}

//------------------------------------------------------------------------

template <class T> T Set<T>::replace(const T& value)
{
    int slot = findSlot(value, hash<T>(value) >> 1, false);
    FW_ASSERT(m_hashes[slot] >= 0);

    T old = m_values[slot];
    m_values[slot] = value;
    return old;
}

//------------------------------------------------------------------------

template <class T> int Set<T>::nextSlot(int slot) const
{
    FW_ASSERT(slot >= -1 && slot < m_capacity);
    for (slot++; slot < m_capacity; slot++)
        if (m_hashes[slot] >= 0)
            return slot;
    return -1;
}

//------------------------------------------------------------------------

template <class T> int Set<T>::findSlot(const T& value, S32 hashValue, bool needEmpty) const
{
    FW_ASSERT(hashValue >= 0);
    if (!m_capacity)
        return -1;

    int blockMask   = (m_capacity - 1) & -BlockSize;
    int firstSlot   = hashValue;
    int firstBlock  = firstSlot & blockMask;
    int blockStep   = BlockSize * 3 + ((hashValue >> 17) & (-4 * BlockSize));

    int block = firstBlock;
    do
    {
        if (needEmpty)
        {
            for (int i = 0; i < BlockSize; i++)
            {
                int slot = block + ((firstSlot + i) & (BlockSize - 1));
                if (m_hashes[slot] < 0)
                    return slot;
            }
        }
        else
        {
            for (int i = 0; i < BlockSize; i++)
            {
                int slot = block + ((firstSlot + i) & (BlockSize - 1));
                S32 slotHash = m_hashes[slot];

                if (slotHash == Empty)
                    return -1;

                if (slotHash == hashValue && equals<T>(m_values[slot], value))
                    return slot;
            }
        }

        block = (block + blockStep) & blockMask;
        blockStep += BlockSize * 4;
    }
    while (block != firstBlock);
    return -1;
}

//------------------------------------------------------------------------

template <class T> void Set<T>::rehash(int capacity)
{
    FW_ASSERT(capacity >= BlockSize);
    FW_ASSERT(capacity >= m_numItems);

    int oldCapacity = m_capacity;
    S32* oldHashes  = m_hashes;
    T* oldValues    = m_values;
    m_capacity      = capacity;
    m_numNonEmpty   = m_numItems;
    m_hashes        = new S32[capacity];
    m_values        = new T[capacity];

    memset(m_hashes, Empty, capacity * sizeof(S32));

    for (int i = 0; i < oldCapacity; i++)
    {
        S32 oldHash = oldHashes[i];
        if (oldHash < 0)
            continue;

        const T& oldValue = oldValues[i];
        int slot = findSlot(oldValue, oldHash, true);
        FW_ASSERT(m_hashes[slot] == Empty);

        m_hashes[slot] = oldHash;
        m_values[slot] = oldValue;
    }

    delete[] oldHashes;
    delete[] oldValues;
}

//------------------------------------------------------------------------

template <class T> bool equalsArray(const T* ptrA, int sizeA, const T* ptrB, int sizeB)
{
    if (sizeA != sizeB)
        return false;

    for (int i = 0; i < sizeA; i++)
        if (!equals<T>(ptrA[i], ptrB[i]))
            return false;
    return true;
}

//------------------------------------------------------------------------

template <class T> U32 hashArray(const T* ptr, int size)
{
    FW_ASSERT(size >= 0);
    FW_ASSERT(ptr || !size);

    U32 a = FW_HASH_MAGIC;
    U32 b = FW_HASH_MAGIC;
    U32 c = FW_HASH_MAGIC;

    while (size >= 3)
    {
        a += hash<T>(ptr[0]);
        b += hash<T>(ptr[1]);
        c += hash<T>(ptr[2]);
        FW_JENKINS_MIX(a, b, c);
        ptr += 3;
        size -= 3;
    }

    switch (size)
    {
    case 2: b += hash<T>(ptr[1]);
    case 1: a += hash<T>(ptr[0]);
    }

    c += size;
    FW_JENKINS_MIX(a, b, c);
    return c;
}

//------------------------------------------------------------------------
}
