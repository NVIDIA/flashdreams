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

#include "gui/Window.hpp"
#include "gui/Image.hpp"
#include "gpu/GLContext.hpp"
#include "base/Thread.hpp"

#include <commdlg.h>
#include <ShlObj.h>
#include <ShellAPI.h>

using namespace FW;

//------------------------------------------------------------------------

static const char* const    s_defaultTitle      = "Anonymous window";
static const Vec2i          s_defaultSize       = Vec2i(1024, 768);
static bool                 s_defaultFullScreen = false;
static const char* const    s_windowClassName   = "FrameworkWindowClass";

//------------------------------------------------------------------------

bool            Window::s_inited        = false;
Array<Window*>* Window::s_open          = NULL;
bool            Window::s_ignoreRepaint = false;

//------------------------------------------------------------------------

Window::Window(void)
:   m_glConfigDirty     (false),
    m_gl                (NULL),

    m_isRealized        (false),
    m_isVisible         (true),
	m_enablePaste		(false),

    m_title             (s_defaultTitle),
    m_isFullScreen      (false),
    m_pendingSize       (-1),

    m_pendingKeyFlush   (false),

    m_mouseKnown        (false),
    m_mousePos          (0),
    m_mouseDragCount    (0),
    m_mouseWheelAcc     (0),

    m_prevSize          (-1)
{
    create();
    setSize(s_defaultSize);
    setFullScreen(s_defaultFullScreen);

    FW_ASSERT(s_open);
    s_open->add(this);
}

//------------------------------------------------------------------------

Window::~Window(void)
{
    destroy();

    FW_ASSERT(s_open);
    s_open->removeItem(this);
}

//------------------------------------------------------------------------

void Window::setTitle(const String& title)
{
    if (m_title != title)
    {
        m_title = title;
        SetWindowText(m_hwnd, title.getPtr());
    }
}

//------------------------------------------------------------------------

void Window::setSize(const Vec2i& size)
{
    FW_ASSERT(size.x >= 0 && size.y >= 0);

    if (m_isFullScreen)
    {
        m_pendingSize = size;
        return;
    }

    RECT rc;
    rc.left     = 0;
    rc.top      = 0;
    rc.right    = size.x;
    rc.bottom   = size.y;
    AdjustWindowRect(&rc, GetWindowLong(m_hwnd, GWL_STYLE), (GetMenu(m_hwnd) != NULL));

    SetWindowPos(m_hwnd, NULL, 0, 0, rc.right - rc.left, rc.bottom - rc.top,
        SWP_NOACTIVATE | SWP_NOCOPYBITS | SWP_NOMOVE | SWP_NOZORDER);
}

//------------------------------------------------------------------------

Vec2i Window::getSize(void) const
{
    RECT rect;
    GetClientRect(m_hwnd, &rect);
    return Vec2i(rect.right, rect.bottom);
}

//------------------------------------------------------------------------

void Window::setVisible(bool visible)
{
    if (m_isRealized && m_isVisible != visible)
        ShowWindow(m_hwnd, (visible) ? SW_SHOW : SW_HIDE);
    m_isVisible = visible;
}

//------------------------------------------------------------------------

void Window::setFullScreen(bool isFull)
{
   if (m_isFullScreen == isFull)
        return;

   m_isFullScreen = isFull;
   if (isFull)
   {
        RECT desk;
        m_origStyle = GetWindowLong(m_hwnd, GWL_STYLE);
        GetWindowRect(m_hwnd, &m_origRect);
        GetWindowRect(GetDesktopWindow(), &desk);
        setWindowLong(m_hwnd, GWL_STYLE, (m_origStyle & ~WS_OVERLAPPEDWINDOW) | WS_POPUP);
        SetWindowPos(m_hwnd, HWND_TOP,
            desk.left, desk.top, desk.right - desk.left, desk.bottom - desk.top,
            SWP_NOACTIVATE | SWP_NOCOPYBITS | SWP_NOZORDER);
   }
   else
   {
        setWindowLong(m_hwnd, GWL_STYLE, m_origStyle);
        SetWindowPos(m_hwnd, NULL,
            m_origRect.left,
            m_origRect.top,
            m_origRect.right - m_origRect.left,
            m_origRect.bottom - m_origRect.top,
            SWP_NOACTIVATE | SWP_NOCOPYBITS | SWP_NOZORDER);

        if (m_pendingSize.x != -1)
        {
            setSize(m_pendingSize);
            m_pendingSize = -1;
        }
   }
}

//------------------------------------------------------------------------

void Window::realize(void)
{
    if (!m_isRealized)
    {
        ShowWindow(m_hwnd, (m_isVisible) ? SW_SHOW : SW_HIDE);
        m_isRealized = true;
    }
}

//------------------------------------------------------------------------

void Window::setGLConfig(const GLContext::Config& config)
{
    m_glConfig = config;
    m_glConfigDirty = (m_gl && memcmp(&m_glConfig, &m_gl->getConfig(), sizeof(GLContext::Config)) != 0);
}

//------------------------------------------------------------------------

GLContext* Window::getGL(void)
{
    if (!m_gl)
    {
        m_gl = new GLContext(m_hdc, m_glConfig);
        m_gl->setView(0, getSize());
    }
    m_gl->makeCurrent();
    return m_gl;
}

//------------------------------------------------------------------------

void Window::repaint(void)
{
    InvalidateRect(m_hwnd, NULL, false);
}

//------------------------------------------------------------------------

void Window::repaintNow(void)
{
    if (s_ignoreRepaint)
        return;
    s_ignoreRepaint = true;

    if (m_glConfigDirty)
    {
        m_glConfigDirty = false;
        recreate();
    }

    Vec2i size = getSize();
    if (size.x > 0 && size.y > 0)
    {
        getGL()->setView(0, size);
        if (!getDiscardEvents())
        {
            s_ignoreRepaint = false;
            if (m_prevSize != size)
            {
                m_prevSize = size;
                postEvent(createSimpleEvent(EventType_Resize));
            }
            postEvent(createSimpleEvent(EventType_PrePaint));
            postEvent(createSimpleEvent(EventType_Paint));
            postEvent(createSimpleEvent(EventType_PostPaint));
            s_ignoreRepaint = true;
        }
        getGL()->swapBuffers();
    }

    Thread::yield();
    s_ignoreRepaint = false;
}

//------------------------------------------------------------------------

void Window::requestClose(void)
{
    PostMessage(m_hwnd, WM_CLOSE, 0, 0);
}

//------------------------------------------------------------------------

void Window::enableDrop(bool enable)
{
	DragAcceptFiles(m_hwnd, enable);
}

//------------------------------------------------------------------------

void Window::enablePaste(bool enable)
{
	m_enablePaste = enable;
}

//------------------------------------------------------------------------

void Window::addListener(Listener* listener)
{
    if (!listener || m_listeners.contains(listener))
        return;

    m_listeners.add(listener);
    listener->handleEvent(createSimpleEvent(EventType_AddListener));
}

//------------------------------------------------------------------------

void Window::removeListener(Listener* listener)
{
    if (!m_listeners.contains(listener))
        return;

    m_listeners.removeItem(listener);
    listener->handleEvent(createSimpleEvent(EventType_RemoveListener));
}

//------------------------------------------------------------------------

void Window::removeListeners(void)
{
    while (m_listeners.getSize())
        removeListener(m_listeners.getLast());
}

//------------------------------------------------------------------------

void Window::showMessageDialog(const String& title, const String& text)
{
    bool old = setDiscardEvents(true);
    MessageBox(m_hwnd, text.getPtr(), title.getPtr(), MB_OK);
    setDiscardEvents(old);
}

//------------------------------------------------------------------------

String Window::showFileDialog(const String& title, bool save, const String& filters, const String& initialDir, bool forceInitialDir)
{
    // Form the filter string.

    String filterStr;
    Array<String> extensions;
    int numFilters = 0;
    int start = 0;

    while (start < filters.getLength())
    {
        int colon = filters.indexOf(':', start);
        FW_ASSERT(colon != -1);
        int comma = filters.indexOf(',', colon);
        if (comma == -1)
            comma = filters.getLength();
        FW_ASSERT(colon < comma);

        String all;
        while (start < colon)
        {
            int semi = filters.indexOf(';', start);
            if (semi == -1 || semi > colon)
                semi = colon;

            String ext = filters.substring(start, semi);
            extensions.add(ext);
            start = semi + 1;

            if (all.getLength())
                all += ';';
            all += "*.";
            all += ext;
        }

        String title = filters.substring(colon + 1, comma);
        filterStr.appendf("%s Files (%s)\n%s\n", title.getPtr(), all.getPtr(), all.getPtr());
        numFilters++;
        start = comma + 1;
    }

    // Add "All Supported Formats" and "All Files".

    if (numFilters > 1 && !save)
    {
        String all;
        for (int i = 0; i < extensions.getSize(); i++)
        {
            if (all.getLength())
                all += ';';
            all += "*.";
            all += extensions[i];
        }
        filterStr = sprintf("All Supported Formats (%s)\n%s\n", all.getPtr(), all.getPtr()) + filterStr;
    }
    filterStr += "All Files (*.*)\n*.*\n";

    // Convert linefeeds to null characters.

    Array<char> filterChars(filterStr.getPtr(), filterStr.getLength() + 1);
    for (int i = 0; i < filterChars.getSize(); i++)
        if (filterChars[i] == '\n')
            filterChars[i] = '\0';

    // Setup OPENFILENAME struct.

    char rawPath[MAX_PATH];
    rawPath[0] = '\0';

    U32 flags;
    if (save)
        flags = OFN_CREATEPROMPT | OFN_NOCHANGEDIR | OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST;
    else
        flags = OFN_FILEMUSTEXIST | OFN_HIDEREADONLY | OFN_NOCHANGEDIR;

    OPENFILENAME ofn;
    ofn.lStructSize         = sizeof(ofn);
    ofn.hwndOwner           = m_hwnd;
    ofn.hInstance           = NULL;
    ofn.lpstrFilter         = filterChars.getPtr();
    ofn.lpstrCustomFilter   = NULL;
    ofn.nMaxCustFilter      = 0;
    ofn.nFilterIndex        = 0;
    ofn.lpstrFile           = rawPath;
    ofn.nMaxFile            = FW_ARRAY_SIZE(rawPath);
    ofn.lpstrFileTitle      = NULL;
    ofn.nMaxFileTitle       = 0;
    ofn.lpstrInitialDir     = (initialDir.getLength() && forceInitialDir) ? initialDir.getPtr() : NULL;
    ofn.lpstrTitle          = title.getPtr();
    ofn.Flags               = flags;
    ofn.nFileOffset         = 0;
    ofn.nFileExtension      = 0;
    ofn.lpstrDefExt         = (extensions.getSize()) ? extensions[0].getPtr() : NULL;
    ofn.lCustData           = NULL;
    ofn.lpfnHook            = NULL;
    ofn.lpTemplateName      = NULL;
    ofn.pvReserved          = NULL;
    ofn.dwReserved          = 0;
    ofn.FlagsEx             = 0;

    // Backup current working directory and set it to initialDir.

    char oldCwd[MAX_PATH];
    bool oldCwdValid = (GetCurrentDirectory(FW_ARRAY_SIZE(oldCwd), oldCwd) != 0);
    if (oldCwdValid && initialDir.getLength() && !forceInitialDir)
        SetCurrentDirectory(initialDir.getPtr());

    // Show modal dialog.

    bool old = setDiscardEvents(true);
    bool rawPathValid = (((save) ? GetSaveFileName(&ofn) : GetOpenFileName(&ofn)) != 0);
    setDiscardEvents(old);
    getGL()->swapBuffers();

    // Convert path to absolute and restore current working directory.

    char absolutePath[MAX_PATH];
    bool absolutePathValid = (rawPathValid && GetFullPathName(rawPath, FW_ARRAY_SIZE(absolutePath), absolutePath, NULL) != 0);
    if (oldCwdValid)
        SetCurrentDirectory(oldCwd);

    // Convert path to relative.

    char relativePath[MAX_PATH];
    bool relativePathValid = (oldCwdValid && absolutePathValid &&
        PathRelativePathTo(relativePath, oldCwd, FILE_ATTRIBUTE_DIRECTORY, (absolutePathValid) ? absolutePath : rawPath, 0));

    // Return the best path that we have.

    return (relativePathValid) ? relativePath : (absolutePathValid) ? absolutePath : (rawPathValid) ? rawPath : "";
}

//------------------------------------------------------------------------

void Window::showModalMessage(const String& msg)
{
    if (!m_isRealized || !m_isVisible)
        return;

    GLContext* gl = getGL();
    for (int i = 0; i < 3; i++)
    {
        gl->drawModalMessage(msg);
        gl->swapBuffers();
    }
}

//------------------------------------------------------------------------

void Window::staticInit(void)
{
    if (s_inited)
        return;
    s_inited = true;

    // Register window class.

    WNDCLASS wc;
    wc.style            = 0;
    wc.lpfnWndProc      = DefWindowProc;
    wc.cbClsExtra       = 0;
    wc.cbWndExtra       = 0;
    wc.hInstance        = (HINSTANCE)GetModuleHandle(NULL);
    wc.hIcon            = LoadIcon(wc.hInstance, MAKEINTRESOURCE(101));
    wc.hCursor          = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground    = CreateSolidBrush(RGB(0, 0, 0));
    wc.lpszMenuName     = NULL;
    wc.lpszClassName    = s_windowClassName;

    if (!wc.hIcon)
        wc.hIcon = LoadIcon(NULL, IDI_APPLICATION);

    RegisterClass(&wc);

    // Create list of open windows.

    FW_ASSERT(!s_open);
    s_open = new Array<Window*>;
}

//------------------------------------------------------------------------

void Window::staticDeinit(void)
{
    if (!s_inited)
        return;
    s_inited = false;

    FW_ASSERT(s_open);
    while (s_open->getSize())
        delete s_open->getLast();
    delete s_open;
    s_open = NULL;
}

//------------------------------------------------------------------------

HWND Window::createHWND(void)
{
    staticInit();
    FW_ASSERT(s_open);

    HWND hwnd = CreateWindow(
        s_windowClassName,
        s_defaultTitle,
        WS_OVERLAPPEDWINDOW,
        1, 1, 0, 0,
        NULL,
        NULL,
        (HINSTANCE)GetModuleHandle(NULL),
        NULL);

    if (!hwnd)
        failWin32Error("CreateWindow");

    return hwnd;
}

//------------------------------------------------------------------------

UPTR Window::setWindowLong(HWND hwnd, int idx, UPTR value)
{
#if FW_64
    return (UPTR)SetWindowLongPtr(hwnd, idx, (LONG_PTR)value);
#else
    return (UPTR)SetWindowLong(hwnd, idx, (LONG)value);
#endif
}

//------------------------------------------------------------------------

void Window::realizeAll(void)
{
    if (s_inited)
        for (int i = 0; i < s_open->getSize(); i++)
            s_open->get(i)->realize();
}

//------------------------------------------------------------------------

void Window::pollMessages(void)
{
    bool old = setDiscardEvents(true);

    MSG msg;
    while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    setDiscardEvents(old);
}

//------------------------------------------------------------------------

Window::Event Window::createFileEvent(EventType type, HANDLE hDrop)
{
	Event ev = createSimpleEvent(type);
	for(int idx = 0; ; idx++)
	{
		int len = DragQueryFile((HDROP)hDrop, idx, 0, 0);
		if (!len)
			break;
		char* buf = new char[len+1];
		DragQueryFile((HDROP)hDrop, idx, buf, len+1);
		ev.files.add(String(buf));
		delete[] buf;
	}
	return ev;
}

//------------------------------------------------------------------------

Window::Event Window::createGenericEvent(EventType type, const String& key, S32 chr, bool mouseKnown, const Vec2i& mousePos)
{
    Event ev;
    ev.type             = type;
    ev.key              = key;
    ev.keyUnicode       = keyToUnicode(key);
    ev.chr              = chr;
    ev.mouseKnown       = mouseKnown;
    ev.mousePos         = mousePos;
    ev.mouseDelta       = (mouseKnown && m_mouseKnown) ? mousePos - m_mousePos : 0;
    ev.mouseDragging    = isMouseDragging();
	ev.image			= NULL;
    ev.window           = this;
    return ev;
}

//------------------------------------------------------------------------

void Window::postEvent(const Event& ev)
{
    m_mouseKnown = ev.mouseKnown;
    m_mousePos = ev.mousePos;

    if (ev.type == EventType_KeyDown ||
        ev.type == EventType_KeyUp ||
        ev.type == EventType_Char ||
        ev.type == EventType_Mouse)
    {
        for (int i = m_listeners.getSize() - 1; i >= 0; i--)
            if (hasError() || m_listeners[i]->handleEvent(ev))
                break;
    }
    else
    {
        for (int i = 0; i < m_listeners.getSize(); i++)
            if (hasError() || m_listeners[i]->handleEvent(ev))
                break;
    }

	if (ev.image)
		delete ev.image;

    failIfError();
}

//------------------------------------------------------------------------

void Window::create(void)
{
    m_hwnd = createHWND();
    m_hdc = GetDC(m_hwnd);
    if (!m_hdc)
        failWin32Error("GetDC");

    setWindowLong(m_hwnd, GWLP_USERDATA, (UPTR)this);
    setWindowLong(m_hwnd, GWLP_WNDPROC, (UPTR)staticWindowProc);
}

//------------------------------------------------------------------------

void Window::destroy(void)
{
    delete m_gl;
    setWindowLong(m_hwnd, GWLP_WNDPROC, (UPTR)DefWindowProc);
    ReleaseDC(m_hwnd, m_hdc);
    DestroyWindow(m_hwnd);

    m_gl = NULL;
    m_hdc = NULL;
    m_hwnd = NULL;
}

//------------------------------------------------------------------------

void Window::recreate(void)
{
    // Backup properties.

    RECT rect;
    GetWindowRect(m_hwnd, &rect);
    DWORD style = GetWindowLong(m_hwnd, GWL_STYLE);

    // Recreate.

    destroy();
    create();

    // Restore properties.

    SetWindowText(m_hwnd, m_title.getPtr());
    setWindowLong(m_hwnd, GWL_STYLE, style);
    SetWindowPos(m_hwnd, HWND_TOP, rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top, 0);
    if (m_isRealized)
        ShowWindow(m_hwnd, (m_isVisible) ? SW_SHOW : SW_HIDE);
}

//------------------------------------------------------------------------

LRESULT CALLBACK Window::staticWindowProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    incNestingLevel(1);
    Window* win = (Window*)(LONG_PTR)GetWindowLongPtr(hWnd, GWLP_USERDATA);
    LRESULT res = win->windowProc(uMsg, wParam, lParam);
    incNestingLevel(-1);
    return res;
}

//------------------------------------------------------------------------

LRESULT Window::windowProc(UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    if (uMsg == WM_ACTIVATE || getDiscardEvents())
    {
        if (m_mouseDragCount)
            ReleaseCapture();

        m_pendingKeyFlush   = true;
        m_mouseKnown        = false;
        m_mouseDragCount    = 0;
        m_mouseWheelAcc     = 0;
    }

    if (m_pendingKeyFlush && !getDiscardEvents())
    {
        m_pendingKeyFlush = false;
        for (int slot = m_keysDown.firstSlot(); slot != -1; slot = m_keysDown.nextSlot(slot))
            postEvent(createKeyEvent(false, m_keysDown.getSlot(slot)));
        m_keysDown.clear();
    }

    if (uMsg == WM_ERASEBKGND)
        return 0;

    if (uMsg == WM_PAINT)
    {
        PAINTSTRUCT paint;
        BeginPaint(m_hwnd, &paint);
        EndPaint(m_hwnd, &paint);
        repaintNow();
        return 0;
    }

    if (getDiscardEvents())
        return DefWindowProc(m_hwnd, uMsg, wParam, lParam);

    switch (uMsg)
    {
    case WM_CLOSE:
        postEvent(createSimpleEvent(EventType_Close));
        return 0;

    case WM_KEYDOWN:
    case WM_KEYUP:
    case WM_SYSKEYDOWN:
    case WM_SYSKEYUP:
        {
			// Paste.

			if (m_enablePaste && uMsg == WM_KEYDOWN)
			{
				if ((wParam == 'V' && GetKeyState(VK_CONTROL) < 0) ||
					(wParam == VK_INSERT && GetKeyState(VK_SHIFT) < 0))
				{
					if (OpenClipboard(m_hwnd))
					{
						if (IsClipboardFormatAvailable(CF_HDROP))
						{
							postEvent(createFileEvent(EventType_PasteFiles, (HDROP)GetClipboardData(CF_HDROP)));
						}
						else if (IsClipboardFormatAvailable(CF_DIB))
						{
							BITMAPINFO* bminfo = (BITMAPINFO*)GetClipboardData(CF_DIB);
							BITMAPINFOHEADER& hdr = bminfo->bmiHeader;
							Vec2i size(hdr.biWidth, hdr.biHeight);
							bool flip = (size.y < 0);
							if (flip)
								size.y = -size.y;
							U8*    p   = (U8*)bminfo->bmiColors;
							Image* img = 0;
							if (hdr.biCompression == 0)
							{
								if (hdr.biBitCount == 24)
								{
									// set alpha to 255
									img = new Image(size);
									for (int y=0; y < size.y; y++, p = (U8*)(((U64)p + 3) & ~(U64)3))
									for (int x=0; x < size.x; x++, p += 3)
										img->setABGR(Vec2i(x, flip ? y : (size.y - y - 1)), (p[0] << 16) | (p[1] << 8) | p[2] | 0xff000000u);
								} else if (hdr.biBitCount == 32)
								{
									// untested
									img = new Image(size);
									for (int y=0; y < size.y; y++)
									for (int x=0; x < size.x; x++, p += 4)
										img->setABGR(Vec2i(x, flip ? y : (size.y - y - 1)), (p[0] << 24) | (p[1] << 16) | (p[2] << 8) | p[3]);
								}
							}
							if (img)
							{
								Event ev = createSimpleEvent(EventType_PasteImage);
								ev.image = img;
								postEvent(ev);
							}
						}
						CloseClipboard();
					}
					return 0;
				}
			}

            // Post key event.

            bool down = (uMsg == WM_KEYDOWN || uMsg == WM_SYSKEYDOWN);
            String key = vkeyToKey((U32)wParam);
            if (key.getLength())
            {
                postEvent(createKeyEvent(down, key));
                if (down && !m_keysDown.contains(key))
                    m_keysDown.add(key);
                else if (!down && m_keysDown.contains(key))
                    m_keysDown.remove(key);
            }

            // Post character events.

            BYTE keyState[256];
            GetKeyboardState(keyState);

            WCHAR buf[256];
            int num = ToUnicode(
                (U32)wParam,
                (((U32)lParam >> 16) & 0xff) | ((down) ? 0 : 0x8000),
                keyState,
                buf,
                FW_ARRAY_SIZE(buf),
                0);

            for (int i = 0; i < num; i++)
                postEvent(createCharEvent(buf[i]));
        }
        return 0;

    case WM_MOUSEMOVE:
        {
            // Enable tracking.

            TRACKMOUSEEVENT track;
            track.cbSize        = sizeof(TRACKMOUSEEVENT);
            track.dwFlags       = TME_LEAVE;
            track.hwndTrack     = m_hwnd;
            track.dwHoverTime   = HOVER_DEFAULT;
            TrackMouseEvent(&track);

            // Post event.

            postEvent(createMouseEvent(true, Vec2i((short)LOWORD(lParam), (short)HIWORD(lParam))));
        }
        return 0;

    case WM_MOUSELEAVE:
        if (!m_mouseDragCount)
            postEvent(createMouseEvent(false, 0));
        return 0;

    case WM_LBUTTONDOWN:
    case WM_LBUTTONUP:
    case WM_RBUTTONDOWN:
    case WM_RBUTTONUP:
    case WM_MBUTTONDOWN:
    case WM_MBUTTONUP:
        {
            // Identify the key.

            String key;
            bool down;
            switch (uMsg)
            {
            case WM_LBUTTONDOWN:    key = FW_KEY_MOUSE_LEFT; down = true; break;
            case WM_LBUTTONUP:      key = FW_KEY_MOUSE_LEFT; down = false; break;
            case WM_RBUTTONDOWN:    key = FW_KEY_MOUSE_RIGHT; down = true; break;
            case WM_RBUTTONUP:      key = FW_KEY_MOUSE_RIGHT; down = false; break;
            case WM_MBUTTONDOWN:    key = FW_KEY_MOUSE_MIDDLE; down = true; break;
            case WM_MBUTTONUP:      key = FW_KEY_MOUSE_MIDDLE; down = false; break;
            default:                FW_ASSERT(false); return 0;
            }

            // Update drag status.

            if (down && !m_keysDown.contains(key))
            {
                m_keysDown.add(key);
                if (!m_mouseDragCount)
                    SetCapture(m_hwnd);
                m_mouseDragCount++;
            }
            else if (!down && m_keysDown.contains(key))
            {
                m_keysDown.remove(key);
                m_mouseDragCount--;
                if (!m_mouseDragCount)
                    ReleaseCapture();
            }

            // Post event.

            postEvent(createKeyEvent(down, key));
        }
        return 0;

    case WM_MOUSEWHEEL:
        m_mouseWheelAcc += (short)HIWORD(wParam);
        while (m_mouseWheelAcc >= 120)
        {
            postEvent(createKeyEvent(true, FW_KEY_WHEEL_UP));
            postEvent(createKeyEvent(false, FW_KEY_WHEEL_UP));
            m_mouseWheelAcc -= 120;
        }
        while (m_mouseWheelAcc <= -120)
        {
            postEvent(createKeyEvent(true, FW_KEY_WHEEL_DOWN));
            postEvent(createKeyEvent(false, FW_KEY_WHEEL_DOWN));
            m_mouseWheelAcc += 120;
        }
        return 0;

	case WM_DROPFILES:
		postEvent(createFileEvent(EventType_DropFiles, (HDROP)wParam));
		return 0;

    default:
        break;
    }

    return DefWindowProc(m_hwnd, uMsg, wParam, lParam);
}

//------------------------------------------------------------------------
