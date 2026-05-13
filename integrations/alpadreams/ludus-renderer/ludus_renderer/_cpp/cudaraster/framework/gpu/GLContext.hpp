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
#include "base/DLLImports.hpp"
#include "base/Hash.hpp"

namespace FW
{
//------------------------------------------------------------------------

class Buffer;
class Image;

//------------------------------------------------------------------------

#define FW_GL_SHADER_SOURCE(CODE) #CODE

//------------------------------------------------------------------------

class GLContext
{
public:

    //------------------------------------------------------------------------

    enum FontStyle
    {
        FontStyle_Normal        = 0,
        FontStyle_Bold          = 1 << 0,
        FontStyle_Italic        = 1 << 1,
        FontStyle_BoldItalic    = FontStyle_Bold | FontStyle_Italic,
    };

    //------------------------------------------------------------------------

    struct Config
    {
        S32             numSamples;
        bool            isStereo;

        Config(void)
        {
            numSamples  = 1;
            isStereo    = false;
        }
    };

    //------------------------------------------------------------------------

    class Program
    {
    public:
                        Program         (const String& vertexSource, const String& fragmentSource);

                        Program         (const String& vertexSource,
                                         GLenum geomInputType, GLenum geomOutputType, int geomVerticesOut, const String& geometrySource,
                                         const String& fragmentSource);

                        ~Program        (void);

        GLuint          getHandle       (void) const                { return m_glProgram; }
        GLint           getAttribLoc    (const String& name) const;
        GLint           getUniformLoc   (const String& name) const;

        void            use             (void);

        static GLuint   createGLShader  (GLenum type, const String& typeStr, const String& source);
        static void     linkGLProgram   (GLuint prog);

    private:
        void            init            (const String& vertexSource,
                                         GLenum geomInputType, GLenum geomOutputType, int geomVerticesOut, const String& geometrySource,
                                         const String& fragmentSource);

    private:
                        Program         (const Program&); // forbidden
        Program&        operator=       (const Program&); // forbidden

    private:
        GLuint          m_glVertexShader;
        GLuint          m_glGeometryShader;
        GLuint          m_glFragmentShader;
        GLuint          m_glProgram;
    };

private:

    //------------------------------------------------------------------------

    struct VGVertex
    {
        Vec4f           pos;
        F32             alpha;
    };

    //------------------------------------------------------------------------

    struct TempTexture
    {
        TempTexture*    next;
        TempTexture*    prev;
        Vec2i           size;
        S32             bytes;
        GLuint          handle;
    };

    //------------------------------------------------------------------------

public:
                        GLContext       (HDC hdc, const Config& config = Config());
                        GLContext       (HDC hdc, HGLRC hglrc);
                        ~GLContext      (void);

    const Config&       getConfig       (void) const        { return m_config; }

    void                makeCurrent     (void);
    void                swapBuffers     (void);

    void                setView         (const Vec2i& pos, const Vec2i& size);
    const Vec2i&        getViewPos      (void) const        { return m_viewPos; }
    const Vec2i&        getViewSize     (void) const        { return m_viewSize; }
    const Vec2f&        getViewScale    (void) const        { return m_viewScale; }

    Mat4f               xformFitToView  (const Vec2f& pos, const Vec2f& size) const { return Mat4f::fitToView(pos, size, m_viewSize); }
    Mat4f               xformMatchPixels(void) const        { return Mat4f::translate(Vec3f(-1.0f, -1.0f, 0.0f)) * Mat4f::scale(Vec3f(m_viewScale, 1.0f)); }
    Mat4f               xformMouseToUser(const Mat4f& userToClip) const;

    void                setAttrib       (int loc, int size, GLenum type, int stride, Buffer* buffer, const void* pointer);
    void                setAttrib       (int loc, int size, GLenum type, int stride, const void* pointer) { setAttrib(loc, size, type, stride, NULL, pointer); }
    void                setAttrib       (int loc, int size, GLenum type, int stride, Buffer& buffer, int ofs) { setAttrib(loc, size, type, stride, &buffer, (const void*)(UPTR)ofs); }
    void                resetAttribs    (void);

    void                setUniform      (int loc, S32 v)    { if (loc >= 0) glUniform1i(loc, v); }
    void                setUniform      (int loc, F32 v)    { if (loc >= 0) glUniform1f(loc, v); }
    void                setUniform      (int loc, F64 v)    { if (loc >= 0) glUniform1d(loc, v); }
    void                setUniform      (int loc, const Vec2f& v) { if (loc >= 0) glUniform2f(loc, v.x, v.y); }
    void                setUniform      (int loc, const Vec3f& v) { if (loc >= 0) glUniform3f(loc, v.x, v.y, v.z); }
    void                setUniform      (int loc, const Vec4f& v) { if (loc >= 0) glUniform4f(loc, v.x, v.y, v.z, v.w); }
    void                setUniform      (int loc, const Mat2f& v) { if (loc >= 0) glUniformMatrix2fv(loc, 1, false, v.getPtr()); }
    void                setUniform      (int loc, const Mat3f& v) { if (loc >= 0) glUniformMatrix3fv(loc, 1, false, v.getPtr()); }
    void                setUniform      (int loc, const Mat4f& v) { if (loc >= 0) glUniformMatrix4fv(loc, 1, false, v.getPtr()); }

    const Mat4f&        getVGXform      (void) const        { return m_vgXform; }
    Mat4f               setVGXform      (const Mat4f& m)    { Mat4f old = m_vgXform; m_vgXform = m; return old; }

    void                strokeLine      (const Vec4f& p0, const Vec4f& p1, U32 abgr);
    void                strokeLine      (const Vec2f& p0, const Vec2f& p1, U32 abgr) { strokeLine(Vec4f(p0, 0.0f, 1.0f), Vec4f(p1, 0.0f, 1.0f), abgr); }
    void                fillRect        (const Vec4f& pos, const Vec2f& localSize, const Vec2f& screenSize, U32 abgr);
    void                fillRect        (const Vec2f& pos, const Vec2f& localSize, U32 abgr) { fillRect(Vec4f(pos, 0.0f, 1.0f), localSize, 0.0f, abgr); }
    void                fillRectNS      (const Vec4f& pos, const Vec2f& screenSize, U32 abgr) { fillRect(pos, 0.0f, screenSize, abgr); }
    void                fillRectNS      (const Vec2f& pos, const Vec2f& screenSize, U32 abgr) { fillRect(Vec4f(pos, 0.0f, 1.0f), 0.0f, screenSize, abgr); }
    void                strokeRect      (const Vec4f& pos, const Vec2f& localSize, const Vec2f& screenSize, U32 abgr);
    void                strokeRect      (const Vec2f& pos, const Vec2f& localSize, U32 abgr) { strokeRect(Vec4f(pos, 0.0f, 1.0f), localSize, 0.0f, abgr); }
    void                strokeRectNS    (const Vec4f& pos, const Vec2f& screenSize, U32 abgr) { strokeRect(pos, 0.0f, screenSize, abgr); }
    void                strokeRectNS    (const Vec2f& pos, const Vec2f& screenSize, U32 abgr) { strokeRect(Vec4f(pos, 0.0f, 1.0f), 0.0f, screenSize, abgr); }

    void                setFont         (const String& name, int size, U32 style);
    void                setDefaultFont  (void)              { setFont(s_defaultFontName, s_defaultFontSize, s_defaultFontStyle); }

    int                 getFontHeight   (void) const        { return m_vgFontMetrics.tmHeight; }
    Vec2i               getStringSize   (const String& str);
    Vec2i               drawString      (const String& str, const Vec4f& pos, const Vec2f& align, U32 abgr) { return drawLabel(str, pos, align, abgr, 0); }
    Vec2i               drawString      (const String& str, const Vec2f& pos, const Vec2f& align, U32 abgr) { return drawString(str, Vec4f(pos, 0.0f, 1.0f), align, abgr); }
    Vec2i               drawString      (const String& str, const Vec2f& pos, U32 abgr) { return drawString(str, Vec4f(pos, 0.0f, 1.0f), 0.5f, abgr); }
    Vec2i               drawLabel       (const String& str, const Vec4f& pos, const Vec2f& align, U32 fgABGR, U32 bgABGR);
    Vec2i               drawLabel       (const String& str, const Vec4f& pos, const Vec2f& align, U32 abgr);
    Vec2i               drawLabel       (const String& str, const Vec2f& pos, const Vec2f& align, U32 abgr) { return drawLabel(str, Vec4f(pos, 0.0f, 1.0f), align, abgr); }
    Vec2i               drawLabel       (const String& str, const Vec2f& pos, U32 abgr) { return drawLabel(str, Vec4f(pos, 0.0f, 1.0f), 0.5f, abgr); }
    void                drawModalMessage(const String& msg);

    void                drawImage       (const Image& image, const Vec4f& pos, const Vec2f& align, bool topToBottom = true);
    void                drawImage       (const Image& image, const Vec2f& pos, const Vec2f& align = 0.5f, bool topToBottom = true) { drawImage(image, Vec4f(pos, 0.0f, 1.0f), align, topToBottom); }

    Program*            getProgram      (const String& id) const;
    void                setProgram      (const String& id, Program* prog);

    static void         staticInit      (void);
    static void         staticDeinit    (void);
    static GLContext&   getHeadless     (void)              { staticInit(); FW_ASSERT(s_headless); return *s_headless; }
    static bool         isStereoAvailable(void)             { staticInit(); return s_stereoAvailable; }
    static void         checkErrors     (void);

private:
    static bool         choosePixelFormat(int& formatIdx, HDC hdc, const Config& config);

    void                init            (HDC hdc, HGLRC hglrc);
    void                drawVG          (const VGVertex* vertices, int numVertices, U32 abgr);
    void                setFont         (HFONT font);
    const Vec2i&        uploadString    (const String& str, const Vec2i& strSize); // returns texture size
    void                drawString      (const Vec4f& pos, const Vec2i& strSize, const Vec2i& texSize, const Vec4f& color);
    const Vec2i&        bindTempTexture (const Vec2i& size);
    void                drawTexture     (int unit, const Vec4f& posLo, const Vec2f& posHi, const Vec2f& texLo, const Vec2f& texHi);

private:
                        GLContext       (const GLContext&); // forbidden
    GLContext&          operator=       (const GLContext&); // forbidden

private:
    static const char* const s_defaultFontName;
    static const S32    s_defaultFontSize;
    static const U32    s_defaultFontStyle;

    static bool         s_inited;
    static HWND         s_shareHWND;
    static HDC          s_shareHDC;
    static HGLRC        s_shareHGLRC;
    static GLContext*   s_headless;
    static GLContext*   s_current;
    static bool         s_stereoAvailable;

    static TempTexture  s_tempTextures;
    static Hash<Vec2i, TempTexture*>* s_tempTexHash;
    static S32          s_tempTexBytes;
    static Hash<String, Program*>* s_programs;

    HDC                 m_hdc;
    HDC                 m_memdc;
    HGLRC               m_hglrc;
    Config              m_config;

    Vec2i               m_viewPos;
    Vec2i               m_viewSize;
    Vec2f               m_viewScale;
    S32                 m_numAttribs;

    Mat4f               m_vgXform;
    HFONT               m_vgFont;
    TEXTMETRIC          m_vgFontMetrics;
};

//------------------------------------------------------------------------
}
