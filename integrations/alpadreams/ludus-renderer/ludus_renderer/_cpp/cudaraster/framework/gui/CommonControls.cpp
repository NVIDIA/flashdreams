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

#include "gui/CommonControls.hpp"
#include "gpu/GLContext.hpp"
#include "gui/Image.hpp"
#include "io/File.hpp"
#include "io/StateDump.hpp"

#include <stdio.h>

using namespace FW;

//------------------------------------------------------------------------

static const F32    s_highlightFadeDuration = 3.0f;
static const F32    s_highlightFadeSpeed    = 3.0f;
static const F32    s_initFrameTime         = 1.0f / 60.0f;

//------------------------------------------------------------------------

CommonControls::CommonControls(U32 features)
:   m_features              (features),
    m_window                (NULL),

    m_showControls          (true),
    m_showFPS               (false),

    m_sliderStackBegun      (false),
    m_sliderStackEmpty      (false),
    m_controlVisibility     (true),

    m_viewSize              (1.0f),
    m_fontHeight            (1.0f),
    m_rightX                (0.0f),

    m_dragging              (false),
    m_avgFrameTime          (s_initFrameTime),
    m_screenshot            (false)
{
    clearActive();

    // Query executable path and file name.

    char moduleFullName[256];
    GetModuleFileName(GetModuleHandle(NULL), moduleFullName, sizeof(moduleFullName) - 1);
    moduleFullName[sizeof(moduleFullName) - 1] = '\0';

    // Strip path and extension.

    const char* moduleShortName = moduleFullName;
    for (int i = 0; moduleFullName[i]; i++)
        if (moduleFullName[i] == '/' || moduleFullName[i] == '\\')
            moduleShortName = &moduleFullName[i + 1];
        else if (moduleFullName[i] == '.')
            moduleFullName[i] = '\0';

    // Initialize file prefixes.

    m_stateFilePrefix = sprintf("state_%s_", moduleShortName);
    m_screenshotFilePrefix = sprintf("screenshot_%s_", moduleShortName);
}

//------------------------------------------------------------------------

CommonControls::~CommonControls(void)
{
    resetControls();
}

//------------------------------------------------------------------------

bool CommonControls::handleEvent(const Window::Event& ev)
{
    FW_ASSERT(m_window == ev.window || ev.type == Window::EventType_AddListener);
    bool stopPropagation = false;

    switch (ev.type)
    {
    case Window::EventType_AddListener:
        FW_ASSERT(!m_window);
        m_window = ev.window;
        m_timer.unstart();
        m_dragging = false;
        m_avgFrameTime = s_initFrameTime;
        m_screenshot = false;
        m_window->repaint();
        return false;

    case Window::EventType_RemoveListener:
        m_window->repaint();
        m_window = NULL;
        return false;

    case Window::EventType_KeyDown:
        {
            if (ev.key == FW_KEY_ESCAPE && hasFeature(Feature_CloseOnEsc))
            {
                m_window->requestClose();
            }
            else if (ev.key == FW_KEY_F4 && m_window->isKeyDown(FW_KEY_ALT) && hasFeature(Feature_CloseOnAltF4))
            {
                m_window->requestClose();
            }
            else if (ev.key == FW_KEY_F5 && hasFeature(Feature_RepaintOnF5))
            {
                m_window->repaint();
            }
            else if (ev.key == FW_KEY_F9 && hasFeature(Feature_ShowFPSOnF9))
            {
                m_showFPS = (!m_showFPS);
                m_avgFrameTime = s_initFrameTime;
                m_window->repaint();
            }
            else if (ev.key == FW_KEY_F10 && hasFeature(Feature_HideControlsOnF10))
            {
                m_showControls = (!m_showControls);
                m_dragging = false;
                m_window->repaint();
            }
            else if (ev.key == FW_KEY_F11 && hasFeature(Feature_FullScreenOnF11))
            {
                m_window->toggleFullScreen();
            }
            else if (ev.key == FW_KEY_MOUSE_LEFT && (m_activeSlider != -1 || m_activeToggle != -1))
            {
                m_dragging = true;
                stopPropagation = true;
                if (m_activeToggle != -1)
                    selectToggle(m_toggles[m_activeToggle]);
            }
            else if (ev.key == FW_KEY_MOUSE_MIDDLE && m_activeSlider != -1 && !m_dragging)
            {
                stopPropagation = true;
                enterSliderValue(m_sliders[m_activeSlider]);
            }
            else if (ev.keyUnicode >= '0' && ev.keyUnicode <= '9')
            {
                int idx = ev.keyUnicode - '0';
                if (!m_window->isKeyDown(FW_KEY_ALT) && hasFeature(Feature_LoadStateOnNum) && m_stateObjs.getSize())
                {
                    if (idx == 0)
                        loadStateDialog();
                    else
                        loadState(getStateFileName(idx));
                }
                else if (m_window->isKeyDown(FW_KEY_ALT) && hasFeature(Feature_SaveStateOnAltNum) && m_stateObjs.getSize())
                {
                    if (idx == 0)
                        saveStateDialog();
                    else
                        saveState(getStateFileName(idx));
                }
            }

            // Handle toggles.

            const Key* key = getKey(ev.key);
            for (int i = 0; i < key->toggles.getSize(); i++)
            {
                selectToggle(key->toggles[i]);
                key->toggles[i]->highlightTime = FW_F32_MAX;
            }

            // Handle sliders.

            for (int i = 0; i < key->sliderIncrease.getSize(); i++)
                sliderKeyDown(key->sliderIncrease[i], 1);
            for (int i = 0; i < key->sliderDecrease.getSize(); i++)
                sliderKeyDown(key->sliderDecrease[i], -1);
            if (key->sliderIncrease.getSize() || key->sliderDecrease.getSize())
                m_window->repaint();
        }
        break;

    case Window::EventType_KeyUp:
        if (ev.key == FW_KEY_PRINT_SCREEN && hasFeature(Feature_ScreenshotOnPrtScn))
        {
            m_screenshot = true;
            m_window->repaint();
        }
        else if (ev.key == FW_KEY_MOUSE_LEFT && m_dragging)
        {
            m_dragging = false;
            stopPropagation = true;
        }
        break;

    case Window::EventType_Mouse:
        if (m_dragging && m_activeSlider != -1)
        {
            Slider* s = m_sliders[m_activeSlider];
            setSliderY(s, getSliderY(s, true) - ev.mouseDelta.y);
        }
        break;

    default:
        break;
    }

    GLContext* gl = NULL;
    if (ev.type == Window::EventType_Paint)
    {
        gl = m_window->getGL();
        layout(gl->getViewSize(), (F32)gl->getFontHeight());
    }

    if (!m_dragging)
    {
        clearActive();
        if (m_showControls && ev.mouseKnown && !ev.mouseDragging)
            updateActive(Vec2f((F32)ev.mousePos.x, m_viewSize.y - (F32)ev.mousePos.y - 1.0f));
    }
    if (m_activeSlider != -1)
    {
        m_sliders[m_activeSlider]->highlightTime = FW_F32_MAX;
        if (m_showControls)
            m_window->repaint();
    }
    if (m_activeToggle != -1)
    {
        m_toggles[m_activeToggle]->highlightTime = FW_F32_MAX;
        if (m_showControls)
            m_window->repaint();
    }

    if (ev.type == Window::EventType_Paint)
        render(gl);
    return stopPropagation;
}

//------------------------------------------------------------------------

void CommonControls::message(const String& str, const String& volatileID, U32 abgr)
{
    if (volatileID.getLength())
        for (int i = 0; i < m_messages.getSize(); i++)
            if (m_messages[i].volatileID == volatileID)
            {
                m_messages.remove(i);
                break;
            }

    if (!str.getLength())
        return;

    Message msg;
    msg.string = str;
    msg.highlightTime = FW_F32_MAX;
    msg.volatileID = volatileID;
	msg.abgr = abgr;
    m_messages.insert(0, msg);

    if (m_window)
        m_window->repaint();
}

//------------------------------------------------------------------------

void CommonControls::removeControl(const void* target)
{
    FW_ASSERT(target);

    for (int i = 0; i < m_toggles.getSize(); i++)
    {
        Toggle* t = m_toggles[i];
        if (t->boolTarget != target && t->enumTarget != target)
            continue;

        t->key->toggles.removeItem(t);
        delete m_toggles.remove(i--);
    }

    for (int i = 0; i < m_sliders.getSize(); i++)
    {
        Slider* s = m_sliders[i];
        if (s->floatTarget != target && s->intTarget != target)
            continue;

        if (!s->stackWithPrevious && i < m_sliders.getSize() - 1)
            m_sliders[i + 1]->stackWithPrevious = false;

        s->increaseKey->sliderIncrease.removeItem(s);
        s->decreaseKey->sliderDecrease.removeItem(s);
        delete m_sliders.remove(i--);
    }
}

//------------------------------------------------------------------------

void CommonControls::resetControls(void)
{
    for (int i = 0; i < m_toggles.getSize(); i++)
        delete m_toggles[i];
    m_toggles.clear();

    for (int i = 0; i < m_sliders.getSize(); i++)
        delete m_sliders[i];
    m_sliders.clear();

    for (int slot = m_keyHash.firstSlot(); slot != -1; slot = m_keyHash.nextSlot(slot))
        delete m_keyHash.getSlot(slot).value;
    m_keyHash.clear();

    clearActive();
//    m_dragging = false;
}

//------------------------------------------------------------------------

String CommonControls::getScreenshotFileName(void) const
{
    SYSTEMTIME st;
    FILETIME ft;
    GetSystemTime(&st);
    SystemTimeToFileTime(&st, &ft);
    U64 stamp = *(const U64*)&ft;

    String name = m_screenshotFilePrefix;
    for (int i = 60; i >= 0; i -= 4)
        name += (char)('a' + ((stamp >> i) & 15));
    name += ".png";
    return name;
}

//------------------------------------------------------------------------

bool CommonControls::loadState(const String& fileName)
{
    String oldError = clearError();
    StateDump dump;

    // Read file.
    {
        File file(fileName, File::Read);
        char tag[9];
        file.readFully(tag, 8);
        tag[8] = 0;
        if (String(tag) != "FWState ")
            setError("Invalid state file!");
        if (!hasError())
            file >> dump;
    }

    // Decode state.

    if (!hasError())
    {
        for (int i = 0; i < m_stateObjs.getSize(); i++)
            m_stateObjs[i]->readState(dump);
    }

    // Display status.

    if (hasError())
        message(sprintf("Unable to load state: %s", getError().getPtr()), "StateIO");
    else
        message(sprintf("Loaded state from '%s'", fileName.getPtr()), "StateIO");
    return (!restoreError(oldError));
}

//------------------------------------------------------------------------

bool CommonControls::loadStateDialog(void)
{
    if (!m_window)
        return false;

    String name = m_window->showFileLoadDialog("Load state", "dat:State", ".");
    if (!name.getLength())
        return false;

    return loadState(name);
}

//------------------------------------------------------------------------

bool CommonControls::saveStateDialog(void)
{
    if (!m_window)
        return false;

    String name = m_window->showFileSaveDialog("Save state", "dat:State", ".");
    if (!name.getLength())
        return false;

    return saveState(name);
}

//------------------------------------------------------------------------

bool CommonControls::saveState(const String& fileName)
{
    String oldError = clearError();

    // Encode state.

    StateDump dump;
    for (int i = 0; i < m_stateObjs.getSize(); i++)
        m_stateObjs[i]->writeState(dump);

    // Write file.

    if (!hasError())
    {
        File file(fileName, File::Create);
        file.write("FWState ", 8);
        file << dump;
    }

    // Display status.

    if (hasError())
        message(sprintf("Unable to save state: %s", getError().getPtr()), "StateIO");
    else
        message(sprintf("Saved state to '%s'", fileName.getPtr()), "StateIO");
    return (!restoreError(oldError));
}

//------------------------------------------------------------------------

F32 CommonControls::getKeyBoost(void) const
{
    F32 boost = 1.0f;
    if (m_window)
    {
        if (m_window->isKeyDown(FW_KEY_SPACE))
            boost *= 5.0f;
        if (m_window->isKeyDown(FW_KEY_CONTROL))
            boost /= 5.0f;
    }
    return boost;
}

//------------------------------------------------------------------------

void CommonControls::flashButtonTitles(void)
{
    for (int i = 0; i < m_toggles.getSize(); i++)
        m_toggles[i]->highlightTime = FW_F32_MAX;
    if (m_window)
        m_window->repaint();
}

//------------------------------------------------------------------------

void CommonControls::render(GLContext* gl)
{
    FW_ASSERT(gl);
    Mat4f oldVGXform = gl->setVGXform(gl->xformMatchPixels());
    glPushAttrib(GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    bool showControls = m_showControls;
    bool repaint = false;

    // Screenshot requested => execute.

    if (m_screenshot)
    {
        // Capture image.

        const Vec2i& size = gl->getViewSize();
        Image image(size, ImageFormat::R8_G8_B8_A8);
        glUseProgram(0);
        glWindowPos2i(0, 0);
        glReadPixels(0, 0, size.x, size.y, GL_RGBA, GL_UNSIGNED_BYTE, image.getMutablePtr());

        // Display the captured image immediately.

        for (int i = 0; i < 3; i++)
        {
            glDrawPixels(size.x, size.y, GL_RGBA, GL_UNSIGNED_BYTE, image.getPtr());
            gl->swapBuffers();
        }
        glDrawPixels(size.x, size.y, GL_RGBA, GL_UNSIGNED_BYTE, image.getPtr());
        showControls = false;
        repaint = true;

        // Export.

        String name = getScreenshotFileName();
        image.flipY();
        exportImage(name, &image);
        message(sprintf("Saved screenshot to '%s'", name.getPtr()));
        m_screenshot = false;
    }

    // Advace time.

    F32 timeDelta = m_timer.end();

    // Draw toggles.

    for (int i = 0; i < m_toggles.getSize(); i++)
    {
        Toggle* t = m_toggles[i];
        F32 fade = updateHighlightFade(&t->highlightTime);
        bool down = (!t->isButton || (m_activeToggle == i && m_dragging));
        if (!showControls || !t->visible || t->isSeparator)
            continue;

        drawPanel(gl, t->pos + t->size * 0.5f, t->size,
            0x80808080, (down) ? 0x80000000 : 0x80FFFFFF, (down) ? 0x80FFFFFF : 0x80000000);

        if (t->isButton)
            drawPanel(gl, t->pos + t->size * 0.5f, t->size - 2.0f,
                0x80808080, (down) ? 0xFF000000 : 0xFFFFFFFF, (down) ? 0xFFFFFFFF : 0xFF000000);

        if (!t->isButton)
        {
            if ((t->boolTarget && *t->boolTarget) ||
                (t->enumTarget && *t->enumTarget == t->enumValue))
            {
                drawPanel(gl, t->pos + t->size * 0.5f, t->size - 4.0f,
                    0x80808080, 0xFFFFFFFF, 0xFF000000);
            }
        }

        if (fade < 1.0f && m_activeToggle != i)
        {
            gl->drawLabel(t->title,
                t->pos + Vec2f(-5.0f, t->size.y * 0.5f), Vec2f(1.0f, 0.5f),
                fadeABGR(0xFFFFFFFF, fade));

            repaint = true;
        }
    }

    // Draw sliders.

    for (int i = 0; i < m_sliders.getSize(); i++)
    {
        Slider* s = m_sliders[i];
        F32 delta = 0.0f;
        if (m_window->isKeyDown(s->increaseKey->id))
            delta += 1.0f;
        if (m_window->isKeyDown(s->decreaseKey->id))
            delta -= 1.0f;
        delta *= getKeyBoost();

        if (delta != 0.0f)
        {
            setSliderValue(s, getSliderValue(s, true) + delta * timeDelta * s->speed);
            s->highlightTime = FW_F32_MAX;
            repaint = true;
        }

        F32 fade = updateHighlightFade(&s->highlightTime);
        bool down = (m_activeSlider == i && m_dragging);
        if (!showControls || !s->visible)
            continue;

        drawPanel(gl, s->pos + s->size * 0.5f, s->size,
            0x80808080, 0x80000000, 0x80FFFFFF);

        drawPanel(gl, s->blockPos + s->blockSize * 0.5f, s->blockSize,
            0x80808080, (down) ? 0xFF000000 : 0xFFFFFFFF, (down) ? 0xFFFFFFFF : 0xFF000000);

        if (fade < 1.0f && m_activeSlider != i)
        {
            gl->drawLabel(getSliderLabel(s),
                Vec2f(s->pos.x - 4.0f, s->blockPos.y + s->blockSize.y * 0.5f),
                Vec2f(1.0f, 0.5f), fadeABGR(0xFFFFFFFF, fade));
            repaint = true;
        }
    }

    // Update and draw FPS counter.

    if (m_showFPS)
    {
        m_avgFrameTime = lerp(timeDelta, m_avgFrameTime, exp2(-timeDelta / 0.3f));
        if (showControls)
            gl->drawLabel(sprintf("%.2f FPS", 1.0f / m_avgFrameTime),
                Vec2f(m_rightX - 4.0f, m_viewSize.y - 2.0f),
                Vec2f(1.0f, 1.0f), 0xFFFFFFFF);
        repaint = true;
    }

    // Draw messages.

    F32 messageY = 2.0f;
    for (int i = 0; i < m_messages.getSize(); i++)
    {
        Message& msg = m_messages[i];
        F32 fade = updateHighlightFade(&msg.highlightTime);

        if (fade >= 1.0f || (i && messageY >= m_viewSize.y))
        {
            m_messages.resize(i);
            break;
        }

        if (showControls)
            gl->drawLabel(msg.string, Vec2f(4.0f, messageY), 0.0f, fadeABGR(msg.abgr, fade));
        repaint = true;
        messageY += m_fontHeight;
    }

    // Draw active controls.

    if (showControls)
    {
        if (m_activeToggle != -1)
        {
            Toggle* t = m_toggles[m_activeToggle];
            if (t->visible)
            {
            gl->drawLabel(t->title,
                t->pos + Vec2f(-5.0f, t->size.y * 0.5f), Vec2f(1.0f, 0.5f), 0xFFFFFFFF);
            repaint = true;
        }
        }
        if (m_activeSlider != -1)
        {
            Slider* s = m_sliders[m_activeSlider];
            if (s->visible)
            {
            gl->drawLabel(getSliderLabel(s),
                Vec2f(s->pos.x - 4.0f, s->blockPos.y + s->blockSize.y * 0.5f), Vec2f(1.0f, 0.5f), 0xFFFFFFFF);
            repaint = true;
        }
    }
    }

    // Finish up.

    gl->setVGXform(oldVGXform);
    glPopAttrib();
    if (repaint)
        m_window->repaint();
    else
        m_timer.unstart();
}

//------------------------------------------------------------------------

void CommonControls::addToggle(bool* boolTarget, S32* enumTarget, S32 enumValue, bool isButton, const String& key, const String& title, bool* dirtyNotify)
{
    Toggle* t           = new Toggle;
    t->boolTarget       = boolTarget;
    t->enumTarget       = enumTarget;
    t->enumValue        = enumValue;
    t->dirtyNotify      = dirtyNotify;

    t->isButton         = isButton;
    t->isSeparator      = (!boolTarget && !enumTarget);
    t->key              = getKey(key);
    t->title            = title;
    t->highlightTime    = -FW_F32_MAX;

    t->visible          = m_controlVisibility;
    t->pos              = Vec2f(0.0f);
    t->size             = Vec2f(0.0f);

    m_toggles.add(t);
    t->key->toggles.add(t);

    if (m_showControls && m_window)
        m_window->repaint();
}

//------------------------------------------------------------------------

void CommonControls::addSlider(F32* floatTarget, S32* intTarget, F32 minValue, F32 maxValue, bool isExponential, const String& increaseKey, const String& decreaseKey, const String& format, F32 speed, bool* dirtyNotify)
{
    FW_ASSERT(floatTarget || intTarget);

    Slider* s           = new Slider;
    s->floatTarget      = floatTarget;
    s->intTarget        = intTarget;
    s->dirtyNotify      = dirtyNotify;

    s->slack            = 0.0f;
    s->minValue         = minValue;
    s->maxValue         = maxValue;
    s->isExponential    = isExponential;
    s->increaseKey      = getKey(increaseKey);
    s->decreaseKey      = getKey(decreaseKey);
    s->format           = format;
    s->speed            = speed;
    s->highlightTime    = -FW_F32_MAX;

    s->visible          = m_controlVisibility;
    s->stackWithPrevious = (m_sliderStackBegun && !m_sliderStackEmpty);
    s->pos              = Vec2f(0.0f);
    s->size             = Vec2f(0.0f);
    s->blockPos         = Vec2f(0.0f);
    s->blockSize        = Vec2f(0.0f);

    m_sliders.add(s);
    s->increaseKey->sliderIncrease.add(s);
    s->decreaseKey->sliderDecrease.add(s);

    m_sliderStackEmpty = false;

    if (m_showControls && m_window)
        m_window->repaint();
}

//------------------------------------------------------------------------

CommonControls::Key* CommonControls::getKey(const String& id)
{
    Key** found = m_keyHash.search(id);
    if (found)
        return *found;

    Key* key = new Key;
    key->id = id;
    m_keyHash.add(id, key);
    return key;
}

//------------------------------------------------------------------------

void CommonControls::layout(const Vec2f& viewSize, F32 fontHeight)
{
    // Setup metrics.

    m_viewSize = viewSize;
    m_fontHeight = fontHeight;
    m_rightX = m_viewSize.x;

    // Layout sliders.

    F32 sliderW = m_fontHeight;
    for (int startSlider = m_sliders.getSize() - 1; startSlider >= 0; startSlider--)
    {
        int endSlider = startSlider + 1;
        while (startSlider && m_sliders[startSlider]->stackWithPrevious)
            startSlider--;

        int numVisible = 0;
        for (int i = startSlider; i < endSlider; i++)
            if (m_sliders[i]->visible)
                numVisible++;
        if (!numVisible)
            continue;

        m_rightX -= sliderW;
        F32 sliderY = 0.0f;
        for (int i = endSlider - 1; i >= startSlider; i--)
        {
            Slider* s    = m_sliders[i];
            if (!s->visible)
                continue;

            s->size      = Vec2f(sliderW, max(floor((viewSize.y - sliderY) / (F32)numVisible + 0.5f), 4.0f));
            s->pos       = Vec2f(m_rightX, sliderY);
            s->blockSize = Vec2f(sliderW - 2.0f, min((m_fontHeight - 2.0f) * 2.0f, s->size.y * 0.25f));
            s->blockPos  = Vec2f(m_rightX + 1.0f, getSliderY(s, false));
            sliderY      += s->size.y;
            numVisible--;
        }
    }

    // Layout toggles.

    Vec2f toggleSize = Vec2f(m_fontHeight + 2.0f);
    int maxInCol = max((int)(viewSize.y / toggleSize.y), 1);
    int numInCol = 0;
    int endIdx = m_toggles.getSize();

    while (endIdx > 0)
    {
        while (endIdx > 0 && (!m_toggles[endIdx - 1]->visible || m_toggles[endIdx - 1]->isSeparator))
            endIdx--;

        int startIdx;
        int numInStrip = 0;
        for (startIdx = endIdx; startIdx > 0; startIdx--)
        {
            const Toggle* t = m_toggles[startIdx - 1];
            if (!t->visible)
                continue;
            if (t->isSeparator)
                break;

            numInStrip++;
            if (numInCol + numInStrip > maxInCol)
        {
            numInCol = 0;
                if (numInStrip > maxInCol)
                    break;
            }
        }

        for (int i = endIdx - 1; i >= startIdx; i--)
        {
            Toggle* t = m_toggles[i];
            if (!t->visible)
                continue;

            if (!numInCol)
                m_rightX -= toggleSize.x;
            t->pos = Vec2f(m_rightX, numInCol * toggleSize.y);
            t->size = toggleSize;
            numInCol++;
        }
        endIdx = startIdx;
        numInCol++;
    }
}

//------------------------------------------------------------------------

void CommonControls::clearActive(void)
{
    m_activeToggle = -1;
    m_activeSlider = -1;
}

//------------------------------------------------------------------------

void CommonControls::updateActive(const Vec2f& mousePos)
{
    clearActive();

    for (int i = 0; i < m_toggles.getSize(); i++)
    {
        Toggle* t = m_toggles[i];
        if (t->visible && !t->isSeparator &&
            mousePos.x >= t->pos.x && mousePos.x < t->pos.x + t->size.x &&
            mousePos.y >= t->pos.y && mousePos.y < t->pos.y + t->size.y)
        {
            m_activeToggle = i;
            return;
        }
    }

    for (int i = 0; i < m_sliders.getSize(); i++)
    {
        Slider* s = m_sliders[i];
        if (s->visible &&
            mousePos.x >= s->blockPos.x && mousePos.x < s->blockPos.x + s->blockSize.x &&
            mousePos.y >= s->blockPos.y && mousePos.y < s->blockPos.y + s->blockSize.y)
        {
            m_activeSlider = i;
            return;
        }
    }
}

//------------------------------------------------------------------------

F32 CommonControls::updateHighlightFade(F32* highlightTime)
{
    *highlightTime = min(*highlightTime, m_timer.getTotal());
    return (m_timer.getTotal() - *highlightTime) / s_highlightFadeDuration;
}

//------------------------------------------------------------------------

U32 CommonControls::fadeABGR(U32 abgr, F32 fade)
{
    Vec4f color = Vec4f::fromABGR(abgr);
    color.w *= min((1.0f - fade) * s_highlightFadeSpeed, 1.0f);
    return color.toABGR();
}

//------------------------------------------------------------------------

void CommonControls::selectToggle(Toggle* t)
{
    FW_ASSERT(t);

    if (t->boolTarget)
        *t->boolTarget = (!*t->boolTarget);

    if (t->enumTarget)
    {
        if (*t->enumTarget == t->enumValue)
            return;
        *t->enumTarget = t->enumValue;
    }

    if (t->dirtyNotify)
        *t->dirtyNotify = true;
}

//------------------------------------------------------------------------

F32 CommonControls::getSliderY(const Slider* s, bool applySlack) const
{
    return getSliderValue(s, applySlack) * (s->size.y - s->blockSize.y - 3.0f) + s->pos.y + 2.0f;
}

//------------------------------------------------------------------------

void CommonControls::setSliderY(Slider* s, F32 y)
{
    setSliderValue(s, (y - s->pos.y - 2.0f) / (s->size.y - s->blockSize.y - 3.0f));
}

//------------------------------------------------------------------------

F32 CommonControls::getSliderValue(const Slider* s, bool applySlack)
{
    FW_ASSERT(s);
    F32 raw = (s->floatTarget) ? *s->floatTarget : (F32)*s->intTarget;
    F32 slacked = raw + ((applySlack) ? s->slack : 0.0f);

    F32 relative = (s->isExponential) ?
        (log(slacked) - log(s->minValue)) / (log(s->maxValue) - log(s->minValue)) :
        (slacked - s->minValue) / (s->maxValue - s->minValue);

    return clamp(relative, 0.0f, 1.0f);
}

//------------------------------------------------------------------------

void CommonControls::setSliderValue(Slider* s, F32 v)
{
    FW_ASSERT(s);
    F32 clamped = clamp(v, 0.0f, 1.0f);

    F32 raw = (s->isExponential) ?
        exp(lerp(log(s->minValue), log(s->maxValue), clamped)) :
        lerp(s->minValue, s->maxValue, clamped);

    bool dirty;
    if (s->floatTarget)
    {
        dirty = (*s->floatTarget != raw);
        *s->floatTarget = raw;
        s->slack = 0.0f;
    }
    else
    {
        S32 rounded = (S32)(raw + ((raw >= 0.0f) ? 0.5f : -0.5f));
        dirty = (*s->intTarget != rounded);
        *s->intTarget = rounded;
        s->slack = raw - (F32)rounded;
    }

    if (dirty && s->dirtyNotify)
        *s->dirtyNotify = true;
}

//------------------------------------------------------------------------

String CommonControls::getSliderLabel(const Slider* s)
{
    FW_ASSERT(s);
    if (s->floatTarget)
        return sprintf(s->format.getPtr(), *s->floatTarget);
    return sprintf(s->format.getPtr(), *s->intTarget);
}

//------------------------------------------------------------------------

void CommonControls::sliderKeyDown(Slider* s, int dir)
{
    FW_ASSERT(s);
    if (!s->intTarget)
        return;

    S32 oldValue = *s->intTarget;
    *s->intTarget = clamp(*s->intTarget + dir, (S32)s->minValue, (S32)s->maxValue);
    s->slack = 0.0f;

    if (*s->intTarget != oldValue && s->dirtyNotify)
        *s->dirtyNotify = true;
}

//------------------------------------------------------------------------

void CommonControls::enterSliderValue(Slider* s)
{
    FW_ASSERT(s);
    m_window->setVisible(false);
    //m_window.showModalMessage(sprintf("Please enter in text window: %s", s->getSliderLabel().getPtr()));
    printf(sprintf("\nEnter %s:\n", getSliderLabel(s).getPtr()).getPtr());

    // Much like setSliderValue()
    // (FIXME Annoying thing about scanf, if you just hit Enter, it keeps reading)
    bool dirty = false;
    if (s->floatTarget)
    {
        float fval;
        if (scanf_s("%g", &fval))
        {
            dirty = (*s->floatTarget != fval);
            *s->floatTarget = fval;
            s->slack = 0.0f;
        }
        else
        {
            printf("No value entered.\n");
            scanf_s("%*s");             // flush line
        }
    }
    else
    {
        int ival;
        if (scanf_s("%d", &ival))
        {
            dirty = (*s->intTarget != ival);
            *s->intTarget = ival;
            s->slack = 0.0f;
        }
        else
        {
            printf("No value entered.\n");
            scanf_s("%*s");             // flush line
        }
     }

    if (dirty && s->dirtyNotify)
        *s->dirtyNotify = true;

    m_window->setVisible(true);         // FIXME pause if error msg
}

//------------------------------------------------------------------------

void CommonControls::drawPanel(GLContext* gl, const Vec2f& pos, const Vec2f& size, U32 interiorABGR, U32 topLeftABGR, U32 bottomRightABGR)
{
    // Setup vertex attributes.

    Vec2f p = gl->getViewScale();
    Vec2f c = pos * p - 1.0f;
    Vec2f r = (size * 0.5f - 1.0f) * p;
    Vec2f px(p.x, 0.0f);
    Vec2f py(0.0f, p.y);
    Vec2f rx(r.x, 0.0f);
    Vec2f ry(0.0f, r.y);

    Vec2f posAttrib[] =
    {
        c + r, c - rx + ry, c + rx - ry, c - r,
        c - r, c - r - py, c + rx - ry, c + rx - ry + px - py, c + r, c + r + px,
        c + r + px, c + r + p, c - rx + ry, c - rx + ry - px + py, c - r - py, c - r - p,
    };

    Vec4f in = Vec4f::fromABGR(interiorABGR);
    Vec4f tl = Vec4f::fromABGR(topLeftABGR);
    Vec4f br = Vec4f::fromABGR(bottomRightABGR);

    Vec4f colorAttrib[] =
    {
        in, in, in, in,
        br, br, br, br, br, br,
        tl, tl, tl, tl, tl, tl,
    };

    // Create program.

    static const char* progId = "CommonControls::drawPanel";
    GLContext::Program* prog = gl->getProgram(progId);
    if (!prog)
    {
        prog = new GLContext::Program(
            FW_GL_SHADER_SOURCE(
                attribute vec2 posAttrib;
                attribute vec4 colorAttrib;
                varying vec4 colorVarying;
                void main()
                {
                    gl_Position = vec4(posAttrib, 0.0, 1.0);
                    colorVarying = colorAttrib;
                }
            ),
            FW_GL_SHADER_SOURCE(
                varying vec4 colorVarying;
                void main()
                {
                    gl_FragColor = colorVarying;
                }
            ));
        gl->setProgram(progId, prog);
    }

    // Draw.

    prog->use();
    gl->setAttrib(prog->getAttribLoc("posAttrib"), 2, GL_FLOAT, 0, posAttrib);
    gl->setAttrib(prog->getAttribLoc("colorAttrib"), 4, GL_FLOAT, 0, colorAttrib);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, FW_ARRAY_SIZE(posAttrib));
    gl->resetAttribs();
}

//------------------------------------------------------------------------
