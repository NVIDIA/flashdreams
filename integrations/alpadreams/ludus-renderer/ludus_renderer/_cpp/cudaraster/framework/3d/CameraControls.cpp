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

#include "3d/CameraControls.hpp"
#include "3d/Mesh.hpp"
#include "io/StateDump.hpp"

using namespace FW;

//------------------------------------------------------------------------

static const F32 s_mouseRotateSpeed = 0.005f;
static const F32 s_mouseStrafeSpeed = -0.005f;
static const F32 s_keyRotateSpeed   = 1.0f;
static const F32 s_inclinationLimit = 0.2f; // avoid looking directly up or down when aligned

//------------------------------------------------------------------------

CameraControls::CameraControls(CommonControls* commonControls, U32 features)
:   m_commonControls    (commonControls),
    m_features          (features),
    m_window            (NULL),
    m_enableMovement    (true),

    m_dragLeft          (false),
    m_dragMiddle        (false),
    m_dragRight         (false),
    m_alignY            (false),
    m_alignZ            (false)
{
    initDefaults();
    if ((m_features & Feature_StereoControls) != 0 && !GLContext::isStereoAvailable())
        m_features &= ~Feature_StereoControls;
}

//------------------------------------------------------------------------

CameraControls::~CameraControls(void)
{
}

//------------------------------------------------------------------------

bool CameraControls::handleEvent(const Window::Event& ev)
{
    if (!m_commonControls)
        fail("CameraControls attached to a window without CommonControls!");
    CommonControls& cc = *m_commonControls;
    FW_ASSERT(m_window == ev.window || ev.type == Window::EventType_AddListener);

    // Initialize movement.

    Mat3f orient = getOrientation();
    Vec3f rotate = 0.0f;
    Vec3f move   = 0.0f;

    // Handle events.

    switch (ev.type)
    {
    case Window::EventType_AddListener:
        FW_ASSERT(!m_window);
        m_window = ev.window;
        m_timer.unstart();
        m_dragLeft = false;
        m_dragMiddle = false;
        m_dragRight = false;

        cc.addStateObject(this);
        addGUIControls();
        repaint();
        return false;

    case Window::EventType_RemoveListener:
        cc.removeStateObject(this);
        removeGUIControls();
        repaint();
        m_window = NULL;
        return false;

    case Window::EventType_KeyDown:
        if (ev.key == FW_KEY_MOUSE_LEFT)    m_dragLeft = true;
        if (ev.key == FW_KEY_MOUSE_MIDDLE)  m_dragMiddle = true;
        if (ev.key == FW_KEY_MOUSE_RIGHT)   m_dragRight = true;
        if (ev.key == FW_KEY_WHEEL_UP)      m_speed *= 1.2f;
        if (ev.key == FW_KEY_WHEEL_DOWN)    m_speed /= 1.2f;
        break;

    case Window::EventType_KeyUp:
        if (ev.key == FW_KEY_MOUSE_LEFT)    m_dragLeft = false;
        if (ev.key == FW_KEY_MOUSE_MIDDLE)  m_dragMiddle = false;
        if (ev.key == FW_KEY_MOUSE_RIGHT)   m_dragRight = false;
        break;

    case Window::EventType_Mouse:
        {
            Vec3f delta = Vec3f((F32)ev.mouseDelta.x, (F32)-ev.mouseDelta.y, 0.0f);
            if (m_dragLeft)     rotate += delta * s_mouseRotateSpeed;
            if (m_dragMiddle)   move += delta * m_speed * s_mouseStrafeSpeed;
            if (m_dragRight)    move += Vec3f(0.0f, 0.0f, (F32)ev.mouseDelta.y) * m_speed * s_mouseStrafeSpeed;
        }
        break;

    case Window::EventType_Paint:
        {
            F32     timeDelta   = m_timer.end();
            F32     boost       = cc.getKeyBoost();
            Vec3f   rotateTmp   = 0.0f;
            bool    alt         = m_window->isKeyDown(FW_KEY_ALT);

            if (m_window->isKeyDown(FW_KEY_A) || (m_window->isKeyDown(FW_KEY_LEFT) && alt))     move.x -= 1.0f;
            if (m_window->isKeyDown(FW_KEY_D) || (m_window->isKeyDown(FW_KEY_RIGHT) && alt))    move.x += 1.0f;
            if (m_window->isKeyDown(FW_KEY_F) || m_window->isKeyDown(FW_KEY_PAGE_DOWN))         move.y -= 1.0f;
            if (m_window->isKeyDown(FW_KEY_R) || m_window->isKeyDown(FW_KEY_PAGE_UP))           move.y += 1.0f;
            if (m_window->isKeyDown(FW_KEY_W) || (m_window->isKeyDown(FW_KEY_UP) && alt))       move.z -= 1.0f;
            if (m_window->isKeyDown(FW_KEY_S) || (m_window->isKeyDown(FW_KEY_DOWN) && alt))     move.z += 1.0f;

            if (m_window->isKeyDown(FW_KEY_LEFT) && !alt)                                       rotateTmp.x -= 1.0f;
            if (m_window->isKeyDown(FW_KEY_RIGHT) && !alt)                                      rotateTmp.x += 1.0f;
            if (m_window->isKeyDown(FW_KEY_DOWN) && !alt)                                       rotateTmp.y -= 1.0f;
            if (m_window->isKeyDown(FW_KEY_UP) && !alt)                                         rotateTmp.y += 1.0f;
            if (m_window->isKeyDown(FW_KEY_E) || m_window->isKeyDown(FW_KEY_HOME))              rotateTmp.z -= 1.0f;
            if (m_window->isKeyDown(FW_KEY_Q) || m_window->isKeyDown(FW_KEY_INSERT))            rotateTmp.z += 1.0f;

            move *= timeDelta * m_speed * boost;
            rotate += rotateTmp * timeDelta * s_keyRotateSpeed * boost;
        }
        break;

    default:
        break;
    }

    // Apply movement.

    if (m_enableMovement)
    {
    if (!move.isZero())
        m_position += orient * move;

    if (rotate.x != 0.0f || rotate.y != 0.0f)
    {
        Vec3f tmp = orient.col(2) * cos(rotate.x) - orient.col(0) * sin(rotate.x);
        m_forward = (orient.col(1) * sin(rotate.y) - tmp * cos(rotate.y)).normalized();
        if (!m_keepAligned)
            m_up = (orient.col(1) * cos(rotate.y) + tmp * sin(rotate.y)).normalized();
        else if (-m_forward.cross(m_up).dot(tmp.cross(m_up).normalized()) < s_inclinationLimit)
            m_forward = -tmp.normalized();
    }

    if (rotate.z != 0.0f && !m_keepAligned)
    {
        Vec3f up = orient.transposed() * m_up;
        m_up = orient * Vec3f(up.x * cos(rotate.z) - sin(rotate.z), up.x * sin(rotate.z) + up.y * cos(rotate.z), up.z);
    }
    }

    // Apply alignment.

    if (m_alignY)
        m_up = Vec3f(0.0f, 1.0f, 0.0f);
    m_alignY = false;

    if (m_alignZ)
        m_up = Vec3f(0.0f, 0.0f, 1.0f);
    m_alignZ = false;

    // Update stereo mode.

    if (hasFeature(Feature_StereoControls))
    {
        GLContext::Config config = m_window->getGLConfig();
        config.isStereo = (m_enableStereo && GLContext::isStereoAvailable());
        m_window->setGLConfig(config);
    }

    // Repaint continuously.

    if (ev.type == Window::EventType_Paint)
        repaint();
    return false;
}

//------------------------------------------------------------------------

void CameraControls::readState(StateDump& d)
{
    initDefaults();
    d.pushOwner("CameraControls");
    d.get(m_position,           "m_position");
    d.get(m_forward,            "m_forward");
    d.get(m_up,                 "m_up");
    d.get(m_keepAligned,        "m_keepAligned");
    d.get(m_speed,              "m_speed");
    d.get(m_fov,                "m_fov");
    d.get(m_near,               "m_near");
    d.get(m_far,                "m_far");
    d.get(m_enableStereo,       "m_enableStereo");
    d.get(m_stereoSeparation,   "m_stereoSeparation");
    d.get(m_stereoConvergence,  "m_stereoConvergence");
    d.popOwner();
}

//------------------------------------------------------------------------

void CameraControls::writeState(StateDump& d) const
{
    d.pushOwner("CameraControls");
    d.set(m_position,           "m_position");
    d.set(m_forward,            "m_forward");
    d.set(m_up,                 "m_up");
    d.set(m_keepAligned,        "m_keepAligned");
    d.set(m_speed,              "m_speed");
    d.set(m_fov,                "m_fov");
    d.set(m_near,               "m_near");
    d.set(m_far,                "m_far");
    d.set(m_enableStereo,       "m_enableStereo");
    d.set(m_stereoSeparation,   "m_stereoSeparation");
    d.set(m_stereoConvergence,  "m_stereoConvergence");
    d.popOwner();
}

//------------------------------------------------------------------------

Mat3f CameraControls::getOrientation(void) const
{
    Mat3f r;
    r.col(2) = -m_forward.normalized();
    r.col(0) = m_up.cross(r.col(2)).normalized();
    r.col(1) = ((Vec3f)r.col(2)).cross(r.col(0)).normalized();
    return r;
}

//------------------------------------------------------------------------

Mat4f CameraControls::getCameraToWorld(void) const
{
    Mat3f orient = getOrientation();
    Mat4f r;
    r.setCol(0, Vec4f(orient.col(0), 0.0f));
    r.setCol(1, Vec4f(orient.col(1), 0.0f));
    r.setCol(2, Vec4f(orient.col(2), 0.0f));
    r.setCol(3, Vec4f(m_position, 1.0f));
    return r;
}

//------------------------------------------------------------------------

Mat4f CameraControls::getWorldToCamera(void) const
{
    Mat3f orient = getOrientation();
    Vec3f pos = orient.transposed() * m_position;
    Mat4f r;
    r.setRow(0, Vec4f(orient.col(0), -pos.x));
    r.setRow(1, Vec4f(orient.col(1), -pos.y));
    r.setRow(2, Vec4f(orient.col(2), -pos.z));
    return r;
}

//------------------------------------------------------------------------

void CameraControls::setCameraToWorld(const Mat4f& m)
{
    m_position = m.col(3).toCartesian();
    m_forward = -normalize(Vec4f(m.col(2)).getXYZ());

    if (!m_keepAligned)
        m_up = normalize(Vec4f(m.col(1)).getXYZ());
}

//------------------------------------------------------------------------

void CameraControls::initDefaults(void)
{
    m_position          = Vec3f(0.0f, 0.0f, 1.5f);
    m_forward           = Vec3f(0.0f, 0.0f, -1.0f);
    m_up                = Vec3f(0.0f, 1.0f, 0.0f);

    m_keepAligned       = false;
    m_speed             = 0.2f;
    m_fov               = 70.0f;
    m_near              = 0.001f;
    m_far               = 3.0f;

    m_enableStereo      = false;
    m_stereoSeparation  = 0.004f;
    m_stereoConvergence = 0.015f;
}

//------------------------------------------------------------------------

void CameraControls::initForMesh(const MeshBase* mesh)
{
    FW_ASSERT(mesh);
    Vec3f lo, hi;
    mesh->getBBox(lo, hi);

    Vec3f center = (lo + hi) * 0.5f;
    F32 size = (hi - lo).max();
    if (size <= 0.0f)
        return;

    m_position  = center + Vec3f(0.0f, 0.0f, size * 0.75f);
    m_forward   = Vec3f(0.0f, 0.0f, -1.0f);
    m_up        = Vec3f(0.0f, 1.0f, 0.0f);

    m_speed     = size * 0.1f;
    m_near      = size * 0.0005f;
    m_far       = size * 1.5f;

    m_stereoSeparation = size * 0.002f;
}

//------------------------------------------------------------------------

String CameraControls::encodeSignature(void) const
{
    String sig;
    sig += '"';
    encodeFloat(sig,        m_position.x);
    encodeFloat(sig,        m_position.y);
    encodeFloat(sig,        m_position.z);
    encodeDirection(sig,    m_forward);
    encodeDirection(sig,    m_up);
    encodeFloat(sig,        m_speed);
    encodeFloat(sig,        m_fov);
    encodeFloat(sig,        m_near);
    encodeFloat(sig,        m_far);
    encodeBits(sig,         (m_keepAligned) ? 1 : 0);
    sig += "\",";
    return sig;
}

//------------------------------------------------------------------------

void CameraControls::decodeSignature(const String& sig)
{
    const char* src = sig.getPtr();
    while (*src == ' ' || *src == '\t' || *src == '\n') src++;
    if (*src == '"') src++;

    F32   px          = decodeFloat(src);
    F32   py          = decodeFloat(src);
    F32   pz          = decodeFloat(src);
    Vec3f forward     = decodeDirection(src);
    Vec3f up          = decodeDirection(src);
    F32   speed       = decodeFloat(src);
    F32   fov         = decodeFloat(src);
    F32   znear       = decodeFloat(src);
    F32   zfar        = decodeFloat(src);
    bool  keepAligned = (decodeBits(src) != 0);

    if (*src == '"') src++;
    if (*src == ',') src++;
    while (*src == ' ' || *src == '\t' || *src == '\n') src++;
    if (*src)
        setError("CameraControls: Invalid signature!");

    if (hasError())
        return;

    m_position      = Vec3f(px, py, pz);
    m_forward       = forward;
    m_up            = up;
    m_speed         = speed;
    m_fov           = fov;
    m_near          = znear;
    m_far           = zfar;
    m_keepAligned   = keepAligned;
}

//------------------------------------------------------------------------

void CameraControls::addGUIControls(void)
{
    CommonControls& cc = *m_commonControls;

    if (hasFeature(Feature_AlignYButton))
        cc.addButton(&m_alignY, FW_KEY_NONE, "Align camera to Y-axis");
    if (hasFeature(Feature_AlignZButton))
        cc.addButton(&m_alignZ, FW_KEY_NONE, "Align camera to Z-axis");
    if (hasFeature(Feature_KeepAlignToggle))
        cc.addToggle(&m_keepAligned, FW_KEY_NONE, "Retain camera alignment");
    if (hasFeature(Feature_SpeedSlider))
        cc.addSlider(&m_speed, 1.0e-3f, 1.0e4f, true, FW_KEY_PLUS, FW_KEY_MINUS, "Camera speed (+/-, mouse wheel) = %g units/sec", 0.05f);

    cc.beginSliderStack();
    if (hasFeature(Feature_FOVSlider))
        cc.addSlider(&m_fov, 1.0f, 179.0f, false, FW_KEY_NONE, FW_KEY_NONE, "Camera FOV = %.1f degrees", 0.2f);
    if (hasFeature(Feature_NearSlider))
        cc.addSlider(&m_near, 1.0e-3f, 1.0e6f, true, FW_KEY_NONE, FW_KEY_NONE, "Camera near = %g units", 0.05f);
    if (hasFeature(Feature_FarSlider))
        cc.addSlider(&m_far, 1.0e-3f, 1.0e6f, true, FW_KEY_NONE, FW_KEY_NONE, "Camera far = %g units", 0.05f);
    cc.endSliderStack();

    if (hasFeature(Feature_StereoControls))
    {
        cc.addToggle(&m_enableStereo, FW_KEY_NONE, "Enable stereoscopic 3D");
        cc.beginSliderStack();
        cc.addSlider(&m_stereoSeparation, 1.0e-3f, 1.0e3f, true, FW_KEY_NONE, FW_KEY_NONE, "Stereo separation = %g units");
        cc.addSlider(&m_stereoConvergence, 1.0e-4f, 1.0f, true, FW_KEY_NONE, FW_KEY_NONE, "Stereo convergence = %g units");
        cc.endSliderStack();
    }
}

//------------------------------------------------------------------------

void CameraControls::removeGUIControls(void)
{
    CommonControls& cc = *m_commonControls;
    cc.removeControl(&m_keepAligned);
    cc.removeControl(&m_speed);
    cc.removeControl(&m_fov);
    cc.removeControl(&m_near);
    cc.removeControl(&m_far);
    cc.removeControl(&m_enableStereo);
    cc.removeControl(&m_stereoSeparation);
    cc.removeControl(&m_stereoConvergence);
}

//------------------------------------------------------------------------

void CameraControls::encodeBits(String& dst, U32 v)
{
    FW_ASSERT(v < 64);
    int base = (v < 12) ? '/' : (v < 38) ? 'A' - 12 : 'a' - 38;
    dst += (char)(v + base);
}

//------------------------------------------------------------------------

U32 CameraControls::decodeBits(const char*& src)
{
    if (*src >= '/' && *src <= ':') return *src++ - '/';
    if (*src >= 'A' && *src <= 'Z') return *src++ - 'A' + 12;
    if (*src >= 'a' && *src <= 'z') return *src++ - 'a' + 38;
    setError("CameraControls: Invalid signature!");
    return 0;
}

//------------------------------------------------------------------------

void CameraControls::encodeFloat(String& dst, F32 v)
{
    U32 bits = floatToBits(v);
    for (int i = 0; i < 32; i += 6)
        encodeBits(dst, (bits >> i) & 0x3F);
}

//------------------------------------------------------------------------

F32 CameraControls::decodeFloat(const char*& src)
{
    U32 bits = 0;
    for (int i = 0; i < 32; i += 6)
        bits |= decodeBits(src) << i;
    return bitsToFloat(bits);
}

//------------------------------------------------------------------------

void CameraControls::encodeDirection(String& dst, const Vec3f& v)
{
    Vec3f a(abs(v.x), abs(v.y), abs(v.z));
    int axis = (a.x >= max(a.y, a.z)) ? 0 : (a.y >= a.z) ? 1 : 2;

    Vec3f tuv;
    switch (axis)
    {
    case 0:  tuv = v; break;
    case 1:  tuv = Vec3f(v.y, v.z, v.x); break;
    default: tuv = Vec3f(v.z, v.x, v.y); break;
    }

    int face = axis | ((tuv.x >= 0.0f) ? 0 : 4);
    if (tuv.y == 0.0f && tuv.z == 0.0f)
    {
        encodeBits(dst, face | 8);
        return;
    }

    encodeBits(dst, face);
    encodeFloat(dst, tuv.y / abs(tuv.x));
    encodeFloat(dst, tuv.z / abs(tuv.x));
}

//------------------------------------------------------------------------

Vec3f CameraControls::decodeDirection(const char*& src)
{
    int face = decodeBits(src);
    Vec3f tuv;
    tuv.x = ((face & 4) == 0) ? 1.0f : -1.0f;
    tuv.y = ((face & 8) == 0) ? decodeFloat(src) : 0.0f;
    tuv.z = ((face & 8) == 0) ? decodeFloat(src) : 0.0f;
    tuv = tuv.normalized();

    switch (face & 3)
    {
    case 0:  return tuv;
    case 1:  return Vec3f(tuv.z, tuv.x, tuv.y);
    default: return Vec3f(tuv.y, tuv.z, tuv.x);
    }
}

//------------------------------------------------------------------------
