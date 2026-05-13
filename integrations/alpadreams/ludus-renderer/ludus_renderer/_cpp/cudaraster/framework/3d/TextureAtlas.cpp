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

#include "3d/TextureAtlas.hpp"
#include "base/Sort.hpp"

using namespace FW;

//------------------------------------------------------------------------

TextureAtlas::TextureAtlas(const ImageFormat& format)
:   m_format    (format),
    m_atlasSize (0)
{
}

//------------------------------------------------------------------------

TextureAtlas::~TextureAtlas(void)
{
}

//------------------------------------------------------------------------

void TextureAtlas::clear(void)
{
    if (m_items.getSize())
    {
        m_items.reset();
        m_itemHash.clear();
        m_atlasTexture = Texture();
    }
}

//------------------------------------------------------------------------

bool TextureAtlas::addTexture(const Texture& tex, int border, bool wrap)
{
    FW_ASSERT(border >= 0);
    const Image* image = tex.getImage();
    if (!image || min(image->getSize()) <= 0 || m_itemHash.contains(image))
        return false;

    m_itemHash.add(image, m_items.getSize());
    Item& item      = m_items.add();
    item.texture    = tex;
    item.border     = border;
    item.wrap       = wrap;
    item.size       = image->getSize() + border * 2;
    m_atlasTexture  = Texture();
    return true;
}

//------------------------------------------------------------------------

Vec2i TextureAtlas::getTexturePos(const Texture& tex)
{
    validate();
    S32* found = m_itemHash.search(tex.getImage());
    if (!found)
        return 0;

    const Item& item = m_items[*found];
    return item.pos + item.border;
}

//------------------------------------------------------------------------

void TextureAtlas::validate(void)
{
    if (!m_atlasTexture.exists())
    {
        layoutItems();
        if (!hasError())
            createAtlas();
    }
}

//------------------------------------------------------------------------

void TextureAtlas::layoutItems(void)
{
    // Select initial size.

    S32 totalArea = 0;
    for (int i = 0; i < m_items.getSize(); i++)
        totalArea += m_items[i].size.x * m_items[i].size.y;
    Vec2i canvas = (int)sqrt((F32)totalArea);

    // Sort items in descending order of height.

    Array<Vec2i> order(NULL, m_items.getSize());
    for (int i = 0; i < m_items.getSize(); i++)
        order[i] = Vec2i(-m_items[i].size.y, i);
    FW_SORT_ARRAY(order, Vec2i, a.x < b.x || (a.x == b.x && a.y < b.y));

    // Add items iteratively.

    Array<Vec4i> maxRects(Vec4i(0, 0, FW_S32_MAX, FW_S32_MAX));
    for (int itemIdx = 0; itemIdx < m_items.getSize(); itemIdx++)
    {
        Item& item = m_items[order[itemIdx].y];
        const Vec2i& s = item.size;

        // Select position.

        Vec2i bestPos = 0;
        Vec3i bestCost = FW_S32_MAX;
        for (int i = 0; i < maxRects.getSize(); i++)
        {
            const Vec4i& r = maxRects[i];
            if (r.x + s.x > r.z || r.y + s.y > r.w)
                continue;

            Vec3i cost;
            cost.x = max(canvas.x, r.x + s.x) * max(canvas.y, r.y + s.y); // Minimize canvas.
            cost.y = r.y + s.y; // Minimize Y.
            cost.z = r.x + s.x; // Minimize X.

            if (cost.x < bestCost.x || (cost.x == bestCost.x && (cost.y < bestCost.y || (cost.y == bestCost.y && cost.z < bestCost.z))))
            {
                bestPos = r.getXY();
                bestCost = cost;
            }
        }

        item.pos = bestPos;
        canvas = max(canvas, bestPos + s);
        Vec4i t(bestPos, bestPos + s);

        // Split maximal rectangles.

        for (int i = maxRects.getSize() - 1; i >= 0; i--)
        {
            Vec4i r = maxRects[i];
            if (t.x >= r.z || t.y >= r.w || t.z <= r.x || t.w <= r.y)
                continue;

            maxRects.removeSwap(i);
            if (t.x > r.x) maxRects.add(Vec4i(r.x, r.y, t.x, r.w));
            if (t.y > r.y) maxRects.add(Vec4i(r.x, r.y, r.z, t.y));
            if (t.z < r.z) maxRects.add(Vec4i(t.z, r.y, r.z, r.w));
            if (t.w < r.w) maxRects.add(Vec4i(r.x, t.w, r.z, r.w));
        }

        // Remove duplicates.

        for (int i = maxRects.getSize() - 1; i >= 0; i--)
        {
            const Vec4i& a = maxRects[i];
            for (int j = 0; j < maxRects.getSize(); j++)
            {
                const Vec4i& b = maxRects[j];
                if (i != j && a.x >= b.x && a.y >= b.y && a.z <= b.z && a.w <= b.w)
                {
                    maxRects.removeSwap(i);
                    break;
                }
            }
        }
    }

    // Determine final size.

    m_atlasSize = 1;
    for (int i = 0; i < m_items.getSize(); i++)
        m_atlasSize = max(m_atlasSize, m_items[i].pos + m_items[i].size);
}

//------------------------------------------------------------------------

void TextureAtlas::createAtlas(void)
{
    Image* image = new Image(m_atlasSize, m_format);
    m_atlasTexture = Texture(image);
    image->clear();

    for (int i = 0; i < m_items.getSize(); i++)
    {
        const Item& item = m_items[i];
        Vec2i pos = item.pos + item.border;
        Vec2i size = item.size - item.border * 2;
        Vec2i wrap = (item.wrap) ? size - 1 : 0;

        image->set(pos, *item.texture.getImage(), 0, size);
        for (int j = 0; j < item.border; j++)
        {
            image->set(pos + Vec2i(-1, 0), *image, pos + Vec2i(wrap.x, 0), Vec2i(1, size.y));
            image->set(pos + Vec2i(size.x, 0), *image, pos + Vec2i(size.x - 1 - wrap.x, 0), Vec2i(1, size.y));
            image->set(pos + Vec2i(-1, -1), *image, pos + Vec2i(-1, wrap.y), Vec2i(size.x + 2, 1));
            image->set(pos + Vec2i(-1, size.y), *image, pos + Vec2i(-1, size.y - 1 - wrap.y), Vec2i(size.x + 2, 1));
            pos -= 1;
            size += 2;
        }
    }
}

//------------------------------------------------------------------------
