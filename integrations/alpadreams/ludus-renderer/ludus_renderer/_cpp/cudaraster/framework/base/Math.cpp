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

#include "base/Math.hpp"

using namespace FW;

//------------------------------------------------------------------------

Vec4f Vec4f::fromABGR(U32 abgr)
{
    return Vec4f(
        (F32)(abgr & 0xFF) * (1.0f / 255.0f),
        (F32)((abgr >> 8) & 0xFF) * (1.0f / 255.0f),
        (F32)((abgr >> 16) & 0xFF) * (1.0f / 255.0f),
        (F32)(abgr >> 24) * (1.0f / 255.0f));
}

//------------------------------------------------------------------------

U32 Vec4f::toABGR(void) const
{
    return
        ((((U32)(((U64)(FW::clamp(x, 0.0f, 1.0f) * exp2(56)) * 255) >> 55) + 1) >> 1) << 0) |
        ((((U32)(((U64)(FW::clamp(y, 0.0f, 1.0f) * exp2(56)) * 255) >> 55) + 1) >> 1) << 8) |
        ((((U32)(((U64)(FW::clamp(z, 0.0f, 1.0f) * exp2(56)) * 255) >> 55) + 1) >> 1) << 16) |
        ((((U32)(((U64)(FW::clamp(w, 0.0f, 1.0f) * exp2(56)) * 255) >> 55) + 1) >> 1) << 24);
}

//------------------------------------------------------------------------

Mat3f Mat4f::getXYZ(void) const
{
    Mat3f r;
    for (int i = 0; i < 3; i++)
        r.col(i) = Vec4f(col(i)).getXYZ();
    return r;
}

//------------------------------------------------------------------------

Mat4f Mat4f::fitToView(const Vec2f& pos, const Vec2f& size, const Vec2f& viewSize)
{
    FW_ASSERT(size.x != 0.0f && size.y != 0.0f);
    FW_ASSERT(viewSize.x != 0.0f && viewSize.y != 0.0f);

    return
        Mat4f::scale(Vec3f(Vec2f(2.0f) / viewSize, 1.0f)) *
        Mat4f::scale(Vec3f((viewSize / size).min(), 1.0f)) *
        Mat4f::translate(Vec3f(-pos - size * 0.5f, 0.0f));
}

//------------------------------------------------------------------------

Mat4f Mat4f::perspective(F32 fov, F32 nearDist, F32 farDist)
{
	// Camera points towards -z.  0 < near < far.
	// Matrix maps z range [-near, -far] to [-1, 1], after homogeneous division.
    F32 f = rcp(tan(fov * FW_PI / 360.0f));
    F32 d = rcp(nearDist - farDist);

    Mat4f r;
    r.setRow(0, Vec4f(  f,      0.0f,   0.0f,                       0.0f                            ));
    r.setRow(1, Vec4f(  0.0f,   f,      0.0f,                       0.0f                            ));
    r.setRow(2, Vec4f(  0.0f,   0.0f,   (nearDist + farDist) * d,   2.0f * nearDist * farDist * d   ));
    r.setRow(3, Vec4f(  0.0f,   0.0f,   -1.0f,                      0.0f                            ));
    return r;
}

//------------------------------------------------------------------------

Mat3f Mat3f::rotation(const Vec3f& axis, F32 angle)
{
	Mat3f R;
	F32 cosa = cosf(angle);
	F32 sina = sinf(angle);
	R(0,0) = cosa + sqr(axis.x)*(1.0f-cosa);			R(0,1) = axis.x*axis.y*(1.0f-cosa) - axis.z*sina;	R(0,2) = axis.x*axis.z*(1.0f-cosa) + axis.y*sina;
	R(1,0) = axis.x*axis.y*(1.0f-cosa) + axis.z*sina;	R(1,1) = cosa + sqr(axis.y)*(1.0f-cosa);			R(1,2) = axis.y*axis.z*(1.0f-cosa) - axis.x*sina;
	R(2,0) = axis.z*axis.x*(1.0f-cosa) - axis.y*sina;	R(2,1) = axis.z*axis.y*(1.0f-cosa) + axis.x*sina;	R(2,2) = cosa + sqr(axis.z)*(1.0f-cosa);
	return R;
}

Mat3d Mat3d::rotation(const Vec3d& axis, F64 angle)
{
	Mat3d R;
	F64 cosa = cos(angle);
	F64 sina = sin(angle);
	R(0,0) = cosa + sqr(axis.x)*(1.0-cosa);				R(0,1) = axis.x*axis.y*(1.0-cosa) - axis.z*sina;	R(0,2) = axis.x*axis.z*(1.0-cosa) + axis.y*sina;
	R(1,0) = axis.x*axis.y*(1.0-cosa) + axis.z*sina;	R(1,1) = cosa + sqr(axis.y)*(1.0-cosa);				R(1,2) = axis.y*axis.z*(1.0-cosa) - axis.x*sina;
	R(2,0) = axis.z*axis.x*(1.0-cosa) - axis.y*sina;	R(2,1) = axis.z*axis.y*(1.0-cosa) + axis.x*sina;	R(2,2) = cosa + sqr(axis.z)*(1.0-cosa);
	return R;
}
