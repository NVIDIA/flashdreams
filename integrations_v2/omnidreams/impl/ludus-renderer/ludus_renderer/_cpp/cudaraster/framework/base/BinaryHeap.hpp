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

template <class T> class BinaryHeap
{
private:
    struct Item
    {
        T           value;
        S32         slot;       // -1 if the item does not exist
        S32         nextInFreelist; // -1 if last, -2 if not in freelist. Freelist may contain items that are not actually free.
    };

public:
                    BinaryHeap  (void)                          : m_hasBeenBuilt(false), m_freelist(-1) {}
                    BinaryHeap  (const BinaryHeap<T>& other)    { set(other); }
                    ~BinaryHeap (void)                          {}

    int             numIndices  (void) const                    { return m_items.getSize(); } // Maximum idx ever used + 1.
    int             numItems    (void) const                    { return m_slots.getSize(); } // Actual size.
    bool            isEmpty     (void) const                    { return (numItems() == 0); }
    bool            contains    (int idx) const                 { return (idx >= 0 && idx < m_items.getSize() && m_items[idx].slot != -1); }
    const T&        get         (int idx) const                 { FW_ASSERT(contains(idx)); return m_items[idx].value; }

    void            clear       (void)                          { m_items.clear(); m_slots.clear(); m_hasBeenBuilt = false; m_freelist = -1; }
    void            reset       (void)                          { m_items.reset(); m_slots.reset(); m_hasBeenBuilt = false; m_freelist = -1; }
    void            set         (const BinaryHeap<T>& other);
    void            add         (int idx, const T& value);      // Adds or replaces an item with the specified index.
    int             add         (const T& value);               // Allocates a previously unused index.
    T               remove      (int idx);                      // Removes an item with the specified index.

    int             getMinIndex (void);                         // Returns the index of the smallest item.
    const T&        getMin      (void)                          { return get(getMinIndex()); } // Returns the smallest item itself.
    int             removeMinIndex(void)                        { int idx = getMinIndex(); remove(idx); return idx; }
    T               removeMin   (void)                          { return remove(getMinIndex()); }

    const T&        operator[]  (int idx) const                 { return get(idx); }
    BinaryHeap<T>&  operator=   (const BinaryHeap<T>& other)    { set(other); return *this; }

private:
    bool            heapify     (int parentSlot);
    void            adjust      (int idx, bool increased);

private:
    Array<Item>     m_items;
    Array<S32>      m_slots;
    bool            m_hasBeenBuilt;
    S32             m_freelist;
};

//------------------------------------------------------------------------

template <class T> void BinaryHeap<T>::set(const BinaryHeap<T>& other)
{
    m_items         = other.m_items;
    m_slots         = other.m_slots;
    m_hasBeenBuilt  = other.m_hasBeenBuilt;
    m_freelist      = other.m_freelist;
}

//------------------------------------------------------------------------

template <class T> void BinaryHeap<T>::add(int idx, const T& value)
{
    FW_ASSERT(idx >= 0);

    // Ensure that the index exists.

    while (idx >= m_items.getSize())
    {
        Item& item = m_items.add();
        item.slot = -1;
        item.nextInFreelist = m_freelist;
        m_freelist = m_items.getSize() - 1;
    }

    // Replace an existing item or add a new item in the last slot.

    bool increased = false;
    if (m_items[idx].slot != -1)
        increased = (m_items[idx].value < value);
    else
    {
        m_items[idx].slot = m_slots.getSize();
        m_slots.add(idx);
    }
    m_items[idx].value = value;

    // Adjust the slot.

    if (m_hasBeenBuilt)
        adjust(idx, increased);
}

//------------------------------------------------------------------------

template <class T> int BinaryHeap<T>::add(const T& value)
{
    // Allocate an index.

    int idx;
    do
    {
        idx = m_freelist;
        if (idx == -1)
        {
            idx = m_items.getSize();
            break;
        }
        m_freelist = m_items[idx].nextInFreelist;
        m_items[idx].nextInFreelist = -2;
    }
    while (m_items[idx].slot != -1);

    // Add the item.

    add(idx, value);
    return idx;
}

//------------------------------------------------------------------------

template <class T> T BinaryHeap<T>::remove(int idx)
{
    // Not in the heap => ignore.

    if (!contains(idx))
        return T();

    // Will have less than two slots => no need to maintain heap property.

    if (m_slots.getSize() <= 2)
        m_hasBeenBuilt = false;

    // Remove.

    Item item = m_items[idx];
    m_items[idx].slot = -1;

    if (item.nextInFreelist == -2)
    {
        m_items[idx].nextInFreelist = m_freelist;
        m_freelist = idx;
    }

    // Move the last item into the slot.

    int last = m_slots.removeLast();
    if (last != idx)
    {
        m_items[last].slot = item.slot;
        m_slots[item.slot] = last;

        // Adjust the slot.

        if (m_hasBeenBuilt)
            adjust(last, (item.value < m_items[last].value));
    }
    return item.value;
}

//------------------------------------------------------------------------

template <class T> int BinaryHeap<T>::getMinIndex(void)
{
    // Empty => ignore.

    if (isEmpty())
        return -1;

    // Not built and has at least two slots => build now.

    if (!m_hasBeenBuilt && m_slots.getSize() >= 2)
    {
        for (int i = (m_slots.getSize() - 2) >> 1; i >= 0; i--)
        {
            int idx = m_slots[i];
            while (heapify(m_items[idx].slot));
        }
        m_hasBeenBuilt = true;
    }

    // Return the root.

    return m_slots[0];
}

//------------------------------------------------------------------------

template <class T> bool BinaryHeap<T>::heapify(int parentSlot)
{
    FW_ASSERT(parentSlot >= 0 && parentSlot < m_slots.getSize());

    // Find the left-hand child.
    // No children => done.

    int childSlot = (parentSlot << 1) + 1;
    if (childSlot >= m_slots.getSize())
        return false;

    int child = m_slots[childSlot];

    // Right-hand child has a smaller value => use it instead.

    if (childSlot + 1 < m_slots.getSize())
    {
        int other = m_slots[childSlot + 1];
        if (m_items[other].value < m_items[child].value)
        {
            childSlot++;
            child = other;
        }
    }

    // The parent has the smallest value => done.

    int parent = m_slots[parentSlot];
    if (!(m_items[child].value < m_items[parent].value))
        return false;

    /* Swap the parent and child slots. */

    m_items[child].slot = parentSlot;
    m_items[parent].slot = childSlot;
    m_slots[parentSlot] = child;
    m_slots[childSlot] = parent;
    return true;
}

//------------------------------------------------------------------------

template <class T> void BinaryHeap<T>::adjust(int idx, bool increased)
{
    FW_ASSERT(contains(idx));
    if (increased)
        while (heapify(m_items[idx].slot));
    else
        while (m_items[idx].slot != 0 && heapify((m_items[idx].slot - 1) >> 1));
}

//------------------------------------------------------------------------
}
