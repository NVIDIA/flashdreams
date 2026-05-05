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
#include "base/DLLImports.hpp"

#include <math.h>

namespace FW
{
//------------------------------------------------------------------------

FW_CUDA_FUNC F32    sqrt            (F32 a)         { return ::sqrtf(a); }
FW_CUDA_FUNC F64    sqrt            (F64 a)         { return ::sqrt(a); }
FW_CUDA_FUNC S32    abs             (S32 a)         { return (a >= 0) ? a : -a; }
FW_CUDA_FUNC S64    abs             (S64 a)         { return (a >= 0) ? a : -a; }
FW_CUDA_FUNC F32    abs             (F32 a)         { return ::fabsf(a); }
FW_CUDA_FUNC F64    abs             (F64 a)         { return ::abs(a); }
FW_CUDA_FUNC F64    pow             (F64 a, F64 b)  { return ::pow(a, b); }
FW_CUDA_FUNC F64    exp             (F64 a)         { return ::exp(a); }
FW_CUDA_FUNC F64    log             (F64 a)         { return ::log(a); }
FW_CUDA_FUNC F64    sin             (F64 a)         { return ::sin(a); }
FW_CUDA_FUNC F64    cos             (F64 a)         { return ::cos(a); }
FW_CUDA_FUNC F64    tan             (F64 a)         { return ::tan(a); }
FW_CUDA_FUNC F32    asin            (F32 a)         { return ::asinf(a); }
FW_CUDA_FUNC F64    asin            (F64 a)         { return ::asin(a); }
FW_CUDA_FUNC F32    acos            (F32 a)         { return ::acosf(a); }
FW_CUDA_FUNC F64    acos            (F64 a)         { return ::acos(a); }
FW_CUDA_FUNC F32    atan            (F32 a)         { return ::atanf(a); }
FW_CUDA_FUNC F64    atan            (F64 a)         { return ::atan(a); }
FW_CUDA_FUNC F64    atan2           (F64 y, F64 x)  { return ::atan2(y, x); }
FW_CUDA_FUNC F32    atan2           (F32 y, F32 x)  { return ::atan2f(y, x); }
FW_CUDA_FUNC F32    floor           (F32 a)         { return ::floorf(a); }
FW_CUDA_FUNC F64    floor           (F64 a)         { return ::floor(a); }
FW_CUDA_FUNC F32    ceil            (F32 a)         { return ::ceilf(a); }
FW_CUDA_FUNC F64    ceil            (F64 a)         { return ::ceil(a); }
FW_CUDA_FUNC U64    doubleToBits    (F64 a)         { return *(U64*)&a; }
FW_CUDA_FUNC F64    bitsToDouble    (U64 a)         { return *(F64*)&a; }

#if FW_CUDA
FW_CUDA_FUNC F32    pow             (F32 a, F32 b)  { return ::__powf(a, b); }
FW_CUDA_FUNC F32    exp             (F32 a)         { return ::__expf(a); }
FW_CUDA_FUNC F32    exp2            (F32 a)         { return ::exp2f(a); }
FW_CUDA_FUNC F64    exp2            (F64 a)         { return ::exp2(a); }
FW_CUDA_FUNC F32    log             (F32 a)         { return ::__logf(a); }
FW_CUDA_FUNC F32    log2            (F32 a)         { return ::__log2f(a); }
FW_CUDA_FUNC F64    log2            (F64 a)         { return ::log2(a); }
FW_CUDA_FUNC F32    sin             (F32 a)         { return ::__sinf(a); }
FW_CUDA_FUNC F32    cos             (F32 a)         { return ::__cosf(a); }
FW_CUDA_FUNC F32    tan             (F32 a)         { return ::__tanf(a); }
FW_CUDA_FUNC U32    floatToBits     (F32 a)         { return ::__float_as_int(a); }
FW_CUDA_FUNC F32    bitsToFloat     (U32 a)         { return ::__int_as_float(a); }
FW_CUDA_FUNC F32    exp2            (int a)         { return ::exp2f((F32)a); }
FW_CUDA_FUNC F32    fastMin         (F32 a, F32 b)  { return ::fminf(a, b); }
FW_CUDA_FUNC F32    fastMax         (F32 a, F32 b)  { return ::fmaxf(a, b); }
FW_CUDA_FUNC F64    fastMin         (F64 a, F64 b)  { return ::fmin(a, b); }
FW_CUDA_FUNC F64    fastMax         (F64 a, F64 b)  { return ::fmax(a, b); }
#else
inline F32          pow             (F32 a, F32 b)  { return ::powf(a, b); }
inline F32          exp             (F32 a)         { return ::expf(a); }
inline F32          exp2            (F32 a)         { return ::powf(2.0f, a); }
inline F64          exp2            (F64 a)         { return ::pow(2.0, a); }
inline F32          log             (F32 a)         { return ::logf(a); }
inline F32          log2            (F32 a)         { return ::logf(a) / ::logf(2.0f); }
inline F64          log2            (F64 a)         { return ::log(a) / ::log(2.0); }
inline F32          sin             (F32 a)         { return ::sinf(a); }
inline F32          cos             (F32 a)         { return ::cosf(a); }
inline F32          tan             (F32 a)         { return ::tanf(a); }
inline U32          floatToBits     (F32 a)         { return *(U32*)&a; }
inline F32          bitsToFloat     (U32 a)         { return *(F32*)&a; }
inline F32          exp2            (int a)         { return bitsToFloat(clamp(a + 127, 1, 254) << 23); }
inline F32          fastMin         (F32 a, F32 b)  { return (a + b - abs(a - b)) * 0.5f; }
inline F32          fastMax         (F32 a, F32 b)  { return (a + b + abs(a - b)) * 0.5f; }
inline F64          fastMin         (F64 a, F64 b)  { return (a + b - abs(a - b)) * 0.5f; }
inline F64          fastMax         (F64 a, F64 b)  { return (a + b + abs(a - b)) * 0.5f; }
#endif

FW_CUDA_FUNC F32    scale           (F32 a, int b)  { return a * exp2(b); }
FW_CUDA_FUNC int    popc8           (U32 mask);
FW_CUDA_FUNC int    popc16          (U32 mask);
FW_CUDA_FUNC int    popc32          (U32 mask);
FW_CUDA_FUNC int    popc64          (U64 mask);

FW_CUDA_FUNC F32    fastClamp       (F32 v, F32 lo, F32 hi) { return fastMin(fastMax(v, lo), hi); }
FW_CUDA_FUNC F64    fastClamp       (F64 v, F64 lo, F64 hi) { return fastMin(fastMax(v, lo), hi); }

template <class T> FW_CUDA_FUNC T sqr(const T& a) { return a * a; }
template <class T> FW_CUDA_FUNC T rcp(const T& a) { return (a) ? (T)1 / a : (T)0; }
template <class A, class B> FW_CUDA_FUNC A lerp(const A& a, const A& b, const B& t) { return (A)(a * ((B)1 - t) + b * t); }

//------------------------------------------------------------------------

template <class T, int L> class Vector;

template <class T, int L, class S> class VectorBase
{
public:
    FW_CUDA_FUNC                    VectorBase  (void)                      {}

    FW_CUDA_FUNC    const T*        getPtr      (void) const                { return ((S*)this)->getPtr(); }
    FW_CUDA_FUNC    T*              getPtr      (void)                      { return ((S*)this)->getPtr(); }
    FW_CUDA_FUNC    const T&        get         (int idx) const             { FW_ASSERT(idx >= 0 && idx < L); return getPtr()[idx]; }
    FW_CUDA_FUNC    T&              get         (int idx)                   { FW_ASSERT(idx >= 0 && idx < L); return getPtr()[idx]; }
    FW_CUDA_FUNC    T               set         (int idx, const T& a)       { T& slot = get(idx); T old = slot; slot = a; return old; }

    FW_CUDA_FUNC    void            set         (const T& a)                { T* tp = getPtr(); for (int i = 0; i < L; i++) tp[i] = a; }
    FW_CUDA_FUNC    void            set         (const T* ptr)              { FW_ASSERT(ptr); T* tp = getPtr(); for (int i = 0; i < L; i++) tp[i] = ptr[i]; }
    FW_CUDA_FUNC    void            setZero     (void)                      { set((T)0); }

#if !FW_CUDA
                    void            print       (void) const                { const T* tp = getPtr(); for (int i = 0; i < L; i++) printf("%g\n", (F64)tp[i]); }
#endif

    FW_CUDA_FUNC    bool            isZero      (void) const                { const T* tp = getPtr(); for (int i = 0; i < L; i++) if (tp[i] != (T)0) return false; return true; }
    FW_CUDA_FUNC    T               lenSqr      (void) const                { const T* tp = getPtr(); T r = (T)0; for (int i = 0; i < L; i++) r += sqr(tp[i]); return r; }
    FW_CUDA_FUNC    T               length      (void) const                { return sqrt(lenSqr()); }
    FW_CUDA_FUNC    S               normalized  (T len = (T)1) const        { return operator*(len * rcp(length())); }
    FW_CUDA_FUNC    void            normalize   (T len = (T)1)              { set(normalized(len)); }
    FW_CUDA_FUNC    T               min         (void) const                { const T* tp = getPtr(); T r = tp[0]; for (int i = 1; i < L; i++) r = FW::min(r, tp[i]); return r; }
    FW_CUDA_FUNC    T               max         (void) const                { const T* tp = getPtr(); T r = tp[0]; for (int i = 1; i < L; i++) r = FW::max(r, tp[i]); return r; }
    FW_CUDA_FUNC    T               sum         (void) const                { const T* tp = getPtr(); T r = tp[0]; for (int i = 1; i < L; i++) r += tp[i]; return r; }
    FW_CUDA_FUNC    S               abs         (void) const                { const T* tp = getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = FW::abs(tp[i]); return r; }

    FW_CUDA_FUNC    Vector<T, L + 1> toHomogeneous(void) const              { const T* tp = getPtr(); Vector<T, L + 1> r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i]; rp[L] = (T)1; return r; }
    FW_CUDA_FUNC    Vector<T, L - 1> toCartesian(void) const                { const T* tp = getPtr(); Vector<T, L - 1> r; T* rp = r.getPtr(); T c = rcp(tp[L - 1]); for (int i = 0; i < L - 1; i++) rp[i] = tp[i] * c; return r; }

    FW_CUDA_FUNC    const T&        operator[]  (int idx) const             { return get(idx); }
    FW_CUDA_FUNC    T&              operator[]  (int idx)                   { return get(idx); }

    FW_CUDA_FUNC    S&              operator=   (const T& a)                { set(a); return *(S*)this; }
    FW_CUDA_FUNC    S&              operator+=  (const T& a)                { set(operator+(a)); return *(S*)this; }
    FW_CUDA_FUNC    S&              operator-=  (const T& a)                { set(operator-(a)); return *(S*)this; }
    FW_CUDA_FUNC    S&              operator*=  (const T& a)                { set(operator*(a)); return *(S*)this; }
    FW_CUDA_FUNC    S&              operator/=  (const T& a)                { set(operator/(a)); return *(S*)this; }
    FW_CUDA_FUNC    S&              operator%=  (const T& a)                { set(operator%(a)); return *(S*)this; }
    FW_CUDA_FUNC    S&              operator&=  (const T& a)                { set(operator&(a)); return *(S*)this; }
    FW_CUDA_FUNC    S&              operator|=  (const T& a)                { set(operator|(a)); return *(S*)this; }
    FW_CUDA_FUNC    S&              operator^=  (const T& a)                { set(operator^(a)); return *(S*)this; }
    FW_CUDA_FUNC    S&              operator<<= (const T& a)                { set(operator<<(a)); return *(S*)this; }
    FW_CUDA_FUNC    S&              operator>>= (const T& a)                { set(operator>>(a)); return *(S*)this; }

    FW_CUDA_FUNC    S               operator+   (void) const                { return *this; }
    FW_CUDA_FUNC    S               operator-   (void) const                { const T* tp = getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = -tp[i]; return r; }
    FW_CUDA_FUNC    S               operator~   (void) const                { const T* tp = getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = ~tp[i]; return r; }

    FW_CUDA_FUNC    S               operator+   (const T& a) const          { const T* tp = getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] + a; return r; }
    FW_CUDA_FUNC    S               operator-   (const T& a) const          { const T* tp = getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] - a; return r; }
    FW_CUDA_FUNC    S               operator*   (const T& a) const          { const T* tp = getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] * a; return r; }
    FW_CUDA_FUNC    S               operator/   (const T& a) const          { const T* tp = getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] / a; return r; }
    FW_CUDA_FUNC    S               operator%   (const T& a) const          { const T* tp = getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] % a; return r; }
    FW_CUDA_FUNC    S               operator&   (const T& a) const          { const T* tp = getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] & a; return r; }
    FW_CUDA_FUNC    S               operator|   (const T& a) const          { const T* tp = getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] | a; return r; }
    FW_CUDA_FUNC    S               operator^   (const T& a) const          { const T* tp = getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] ^ a; return r; }
    FW_CUDA_FUNC    S               operator<<  (const T& a) const          { const T* tp = getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] << a; return r; }
    FW_CUDA_FUNC    S               operator>>  (const T& a) const          { const T* tp = getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] >> a; return r; }

    template <class V> FW_CUDA_FUNC void    set         (const VectorBase<T, L, V>& v)          { set(v.getPtr()); }
    template <class V> FW_CUDA_FUNC T       dot         (const VectorBase<T, L, V>& v) const    { const T* tp = getPtr(); const T* vp = v.getPtr(); T r = (T)0; for (int i = 0; i < L; i++) r += tp[i] * vp[i]; return r; }
    template <class V> FW_CUDA_FUNC S       min         (const VectorBase<T, L, V>& v) const    { const T* tp = getPtr(); const T* vp = v.getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = FW::min(tp[i], vp[i]); return r; }
    template <class V> FW_CUDA_FUNC S       max         (const VectorBase<T, L, V>& v) const    { const T* tp = getPtr(); const T* vp = v.getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = FW::max(tp[i], vp[i]); return r; }
    template <class V, class W> FW_CUDA_FUNC S clamp    (const VectorBase<T, L, V>& lo, const VectorBase<T, L, W>& hi) const { const T* tp = getPtr(); const T* lop = lo.getPtr(); const T* hip = hi.getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = FW::clamp(tp[i], lop[i], hip[i]); return r; }

    template <class V> FW_CUDA_FUNC S&      operator=   (const VectorBase<T, L, V>& v)          { set(v); return *(S*)this; }
    template <class V> FW_CUDA_FUNC S&      operator+=  (const VectorBase<T, L, V>& v)          { set(operator+(v)); return *(S*)this; }
    template <class V> FW_CUDA_FUNC S&      operator-=  (const VectorBase<T, L, V>& v)          { set(operator-(v)); return *(S*)this; }
    template <class V> FW_CUDA_FUNC S&      operator*=  (const VectorBase<T, L, V>& v)          { set(operator*(v)); return *(S*)this; }
    template <class V> FW_CUDA_FUNC S&      operator/=  (const VectorBase<T, L, V>& v)          { set(operator/(v)); return *(S*)this; }
    template <class V> FW_CUDA_FUNC S&      operator%=  (const VectorBase<T, L, V>& v)          { set(operator%(v)); return *(S*)this; }
    template <class V> FW_CUDA_FUNC S&      operator&=  (const VectorBase<T, L, V>& v)          { set(operator&(v)); return *(S*)this; }
    template <class V> FW_CUDA_FUNC S&      operator|=  (const VectorBase<T, L, V>& v)          { set(operator|(v)); return *(S*)this; }
    template <class V> FW_CUDA_FUNC S&      operator^=  (const VectorBase<T, L, V>& v)          { set(operator^(v)); return *(S*)this; }
    template <class V> FW_CUDA_FUNC S&      operator<<= (const VectorBase<T, L, V>& v)          { set(operator<<(v)); return *(S*)this; }
    template <class V> FW_CUDA_FUNC S&      operator>>= (const VectorBase<T, L, V>& v)          { set(operator>>(v)); return *(S*)this; }

    template <class V> FW_CUDA_FUNC S       operator+   (const VectorBase<T, L, V>& v) const    { const T* tp = getPtr(); const T* vp = v.getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] +  vp[i]; return r; }
    template <class V> FW_CUDA_FUNC S       operator-   (const VectorBase<T, L, V>& v) const    { const T* tp = getPtr(); const T* vp = v.getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] -  vp[i]; return r; }
    template <class V> FW_CUDA_FUNC S       operator*   (const VectorBase<T, L, V>& v) const    { const T* tp = getPtr(); const T* vp = v.getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] *  vp[i]; return r; }
    template <class V> FW_CUDA_FUNC S       operator/   (const VectorBase<T, L, V>& v) const    { const T* tp = getPtr(); const T* vp = v.getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] /  vp[i]; return r; }
    template <class V> FW_CUDA_FUNC S       operator%   (const VectorBase<T, L, V>& v) const    { const T* tp = getPtr(); const T* vp = v.getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] %  vp[i]; return r; }
    template <class V> FW_CUDA_FUNC S       operator&   (const VectorBase<T, L, V>& v) const    { const T* tp = getPtr(); const T* vp = v.getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] &  vp[i]; return r; }
    template <class V> FW_CUDA_FUNC S       operator|   (const VectorBase<T, L, V>& v) const    { const T* tp = getPtr(); const T* vp = v.getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] |  vp[i]; return r; }
    template <class V> FW_CUDA_FUNC S       operator^   (const VectorBase<T, L, V>& v) const    { const T* tp = getPtr(); const T* vp = v.getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] ^  vp[i]; return r; }
    template <class V> FW_CUDA_FUNC S       operator<<  (const VectorBase<T, L, V>& v) const    { const T* tp = getPtr(); const T* vp = v.getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] << vp[i]; return r; }
    template <class V> FW_CUDA_FUNC S       operator>>  (const VectorBase<T, L, V>& v) const    { const T* tp = getPtr(); const T* vp = v.getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] >> vp[i]; return r; }

    template <class V> FW_CUDA_FUNC bool    operator==  (const VectorBase<T, L, V>& v) const    { const T* tp = getPtr(); const T* vp = v.getPtr(); for (int i = 0; i < L; i++) if (tp[i] != vp[i]) return false; return true; }
    template <class V> FW_CUDA_FUNC bool    operator!=  (const VectorBase<T, L, V>& v) const    { return (!operator==(v)); }
};

//------------------------------------------------------------------------

template <class T, int L> class Vector : public VectorBase<T, L, Vector<T, L> >
{
public:
    FW_CUDA_FUNC                    Vector      (void)                      { setZero(); }
    FW_CUDA_FUNC                    Vector      (T a)                       { set(a); }

    FW_CUDA_FUNC    const T*        getPtr      (void) const                { return m_values; }
    FW_CUDA_FUNC    T*              getPtr      (void)                      { return m_values; }
    static FW_CUDA_FUNC Vector      fromPtr     (const T* ptr)              { Vector v; v.set(ptr); return v; }

    template <class V> FW_CUDA_FUNC Vector(const VectorBase<T, L, V>& v) { set(v); }
    template <class V> FW_CUDA_FUNC Vector& operator=(const VectorBase<T, L, V>& v) { set(v); return *this; }

private:
    T               m_values[L];
};

//------------------------------------------------------------------------

class Vec2i : public VectorBase<S32, 2, Vec2i>, public int2
{
public:
    FW_CUDA_FUNC                    Vec2i       (void)                      { setZero(); }
    FW_CUDA_FUNC                    Vec2i       (S32 a)                     { set(a); }
    FW_CUDA_FUNC                    Vec2i       (S32 xx, S32 yy)            { x = xx; y = yy; }

    FW_CUDA_FUNC    const S32*      getPtr      (void) const                { return &x; }
    FW_CUDA_FUNC    S32*            getPtr      (void)                      { return &x; }
    static FW_CUDA_FUNC Vec2i       fromPtr     (const S32* ptr)            { return Vec2i(ptr[0], ptr[1]); }

    FW_CUDA_FUNC    Vec2i           perpendicular(void) const               { return Vec2i(-y, x); }

    template <class V> FW_CUDA_FUNC Vec2i(const VectorBase<S32, 2, V>& v) { set(v); }
    template <class V> FW_CUDA_FUNC Vec2i& operator=(const VectorBase<S32, 2, V>& v) { set(v); return *this; }
};

//------------------------------------------------------------------------

class Vec3i : public VectorBase<S32, 3, Vec3i>, public int3
{
public:
    FW_CUDA_FUNC                    Vec3i       (void)                      { setZero(); }
    FW_CUDA_FUNC                    Vec3i       (S32 a)                     { set(a); }
    FW_CUDA_FUNC                    Vec3i       (S32 xx, S32 yy, S32 zz)    { x = xx; y = yy; z = zz; }
    FW_CUDA_FUNC                    Vec3i       (const Vec2i& xy, S32 zz)   { x = xy.x; y = xy.y; z = zz; }

    FW_CUDA_FUNC    const S32*      getPtr      (void) const                { return &x; }
    FW_CUDA_FUNC    S32*            getPtr      (void)                      { return &x; }
    static FW_CUDA_FUNC Vec3i       fromPtr     (const S32* ptr)            { return Vec3i(ptr[0], ptr[1], ptr[2]); }

    FW_CUDA_FUNC    Vec2i           getXY       (void) const                { return Vec2i(x, y); }

    template <class V> FW_CUDA_FUNC Vec3i(const VectorBase<S32, 3, V>& v) { set(v); }
    template <class V> FW_CUDA_FUNC Vec3i& operator=(const VectorBase<S32, 3, V>& v) { set(v); return *this; }
};

//------------------------------------------------------------------------

class Vec4i : public VectorBase<S32, 4, Vec4i>, public int4
{
public:
    FW_CUDA_FUNC                    Vec4i       (void)                      { setZero(); }
    FW_CUDA_FUNC                    Vec4i       (S32 a)                     { set(a); }
    FW_CUDA_FUNC                    Vec4i       (S32 xx, S32 yy, S32 zz, S32 ww) { x = xx; y = yy; z = zz; w = ww; }
    FW_CUDA_FUNC                    Vec4i       (const Vec2i& xy, S32 zz, S32 ww) { x = xy.x; y = xy.y; z = zz; w = ww; }
    FW_CUDA_FUNC                    Vec4i       (const Vec3i& xyz, S32 ww)  { x = xyz.x; y = xyz.y; z = xyz.z; w = ww; }
    FW_CUDA_FUNC                    Vec4i       (const Vec2i& xy, const Vec2i& zw) { x = xy.x; y = xy.y; z = zw.x; w = zw.y; }

    FW_CUDA_FUNC    const S32*      getPtr      (void) const                { return &x; }
    FW_CUDA_FUNC    S32*            getPtr      (void)                      { return &x; }
    static FW_CUDA_FUNC Vec4i       fromPtr     (const S32* ptr)            { return Vec4i(ptr[0], ptr[1], ptr[2], ptr[3]); }

    FW_CUDA_FUNC    Vec2i           getXY       (void) const                { return Vec2i(x, y); }
    FW_CUDA_FUNC    Vec3i           getXYZ      (void) const                { return Vec3i(x, y, z); }
    FW_CUDA_FUNC    Vec3i           getXYW      (void) const                { return Vec3i(x, y, w); }

    template <class V> FW_CUDA_FUNC Vec4i(const VectorBase<S32, 4, V>& v) { set(v); }
    template <class V> FW_CUDA_FUNC Vec4i& operator=(const VectorBase<S32, 4, V>& v) { set(v); return *this; }
};

//------------------------------------------------------------------------

class Vec2f : public VectorBase<F32, 2, Vec2f>, public float2
{
public:
    FW_CUDA_FUNC                    Vec2f       (void)                      { setZero(); }
    FW_CUDA_FUNC                    Vec2f       (F32 a)                     { set(a); }
    FW_CUDA_FUNC                    Vec2f       (F32 xx, F32 yy)            { x = xx; y = yy; }
    FW_CUDA_FUNC                    Vec2f       (const Vec2i& v)            { x = (F32)v.x; y = (F32)v.y; }

    FW_CUDA_FUNC    const F32*      getPtr      (void) const                { return &x; }
    FW_CUDA_FUNC    F32*            getPtr      (void)                      { return &x; }
    static FW_CUDA_FUNC Vec2f       fromPtr     (const F32* ptr)            { return Vec2f(ptr[0], ptr[1]); }

    FW_CUDA_FUNC    operator        Vec2i       (void) const                { return Vec2i((S32)x, (S32)y); }

    FW_CUDA_FUNC    Vec2f           perpendicular(void) const               { return Vec2f(-y, x); }
    FW_CUDA_FUNC    F32             cross       (const Vec2f& v) const      { return x * v.y - y * v.x; }

    template <class V> FW_CUDA_FUNC Vec2f(const VectorBase<F32, 2, V>& v) { set(v); }
    template <class V> FW_CUDA_FUNC Vec2f& operator=(const VectorBase<F32, 2, V>& v) { set(v); return *this; }
};

//------------------------------------------------------------------------

class Vec3f : public VectorBase<F32, 3, Vec3f>, public float3
{
public:
    FW_CUDA_FUNC                    Vec3f       (void)                      { setZero(); }
    FW_CUDA_FUNC                    Vec3f       (F32 a)                     { set(a); }
    FW_CUDA_FUNC                    Vec3f       (F32 xx, F32 yy, F32 zz)    { x = xx; y = yy; z = zz; }
    FW_CUDA_FUNC                    Vec3f       (const Vec2f& xy, F32 zz)   { x = xy.x; y = xy.y; z = zz; }
    FW_CUDA_FUNC                    Vec3f       (const Vec3i& v)            { x = (F32)v.x; y = (F32)v.y; z = (F32)v.z; }

    FW_CUDA_FUNC    const F32*      getPtr      (void) const                { return &x; }
    FW_CUDA_FUNC    F32*            getPtr      (void)                      { return &x; }
    static FW_CUDA_FUNC Vec3f       fromPtr     (const F32* ptr)            { return Vec3f(ptr[0], ptr[1], ptr[2]); }

    FW_CUDA_FUNC    operator        Vec3i       (void) const                { return Vec3i((S32)x, (S32)y, (S32)z); }
    FW_CUDA_FUNC    Vec2f           getXY       (void) const                { return Vec2f(x, y); }

    FW_CUDA_FUNC    Vec3f           cross       (const Vec3f& v) const      { return Vec3f(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x); }

    template <class V> FW_CUDA_FUNC Vec3f(const VectorBase<F32, 3, V>& v) { set(v); }
    template <class V> FW_CUDA_FUNC Vec3f& operator=(const VectorBase<F32, 3, V>& v) { set(v); return *this; }
};

//------------------------------------------------------------------------

class Vec4f : public VectorBase<F32, 4, Vec4f>, public float4
{
public:
    FW_CUDA_FUNC                    Vec4f       (void)                      { setZero(); }
    FW_CUDA_FUNC                    Vec4f       (F32 a)                     { set(a); }
    FW_CUDA_FUNC                    Vec4f       (F32 xx, F32 yy, F32 zz, F32 ww) { x = xx; y = yy; z = zz; w = ww; }
    FW_CUDA_FUNC                    Vec4f       (const Vec2f& xy, F32 zz, F32 ww) { x = xy.x; y = xy.y; z = zz; w = ww; }
    FW_CUDA_FUNC                    Vec4f       (const Vec3f& xyz, F32 ww)  { x = xyz.x; y = xyz.y; z = xyz.z; w = ww; }
    FW_CUDA_FUNC                    Vec4f       (const Vec2f& xy, const Vec2f& zw) { x = xy.x; y = xy.y; z = zw.x; w = zw.y; }
    FW_CUDA_FUNC                    Vec4f       (const Vec4i& v)            { x = (F32)v.x; y = (F32)v.y; z = (F32)v.z; w = (F32)v.w; }

    FW_CUDA_FUNC    const F32*      getPtr      (void) const                { return &x; }
    FW_CUDA_FUNC    F32*            getPtr      (void)                      { return &x; }
    static FW_CUDA_FUNC Vec4f       fromPtr     (const F32* ptr)            { return Vec4f(ptr[0], ptr[1], ptr[2], ptr[3]); }

    FW_CUDA_FUNC    operator        Vec4i       (void) const                { return Vec4i((S32)x, (S32)y, (S32)z, (S32)w); }
    FW_CUDA_FUNC    Vec2f           getXY       (void) const                { return Vec2f(x, y); }
    FW_CUDA_FUNC    Vec3f           getXYZ      (void) const                { return Vec3f(x, y, z); }
    FW_CUDA_FUNC    Vec3f           getXYW      (void) const                { return Vec3f(x, y, w); }

#if !FW_CUDA
    static Vec4f    fromABGR        (U32 abgr);
    U32             toABGR          (void) const;
#endif

    template <class V> FW_CUDA_FUNC Vec4f(const VectorBase<F32, 4, V>& v) { set(v); }
    template <class V> FW_CUDA_FUNC Vec4f& operator=(const VectorBase<F32, 4, V>& v) { set(v); return *this; }
};

//------------------------------------------------------------------------

class Vec2d : public VectorBase<F64, 2, Vec2d>, public double2
{
public:
    FW_CUDA_FUNC                    Vec2d       (void)                      { setZero(); }
    FW_CUDA_FUNC                    Vec2d       (F64 a)                     { set(a); }
    FW_CUDA_FUNC                    Vec2d       (F64 xx, F64 yy)            { x = xx; y = yy; }
    FW_CUDA_FUNC                    Vec2d       (const Vec2i& v)            { x = (F64)v.x; y = (F64)v.y; }
    FW_CUDA_FUNC                    Vec2d       (const Vec2f& v)            { x = v.x; y = v.y; }

    FW_CUDA_FUNC    const F64*      getPtr      (void) const                { return &x; }
    FW_CUDA_FUNC    F64*            getPtr      (void)                      { return &x; }
    static FW_CUDA_FUNC Vec2d       fromPtr     (const F64* ptr)            { return Vec2d(ptr[0], ptr[1]); }

    FW_CUDA_FUNC    operator        Vec2i       (void) const                { return Vec2i((S32)x, (S32)y); }
    FW_CUDA_FUNC    operator        Vec2f       (void) const                { return Vec2f((F32)x, (F32)y); }

    FW_CUDA_FUNC    Vec2d           perpendicular(void) const               { return Vec2d(-y, x); }
    FW_CUDA_FUNC    F64             cross       (const Vec2d& v) const      { return x * v.y - y * v.x; }

    template <class V> FW_CUDA_FUNC Vec2d(const VectorBase<F64, 2, V>& v) { set(v); }
    template <class V> FW_CUDA_FUNC Vec2d& operator=(const VectorBase<F64, 2, V>& v) { set(v); return *this; }
};

//------------------------------------------------------------------------

class Vec3d : public VectorBase<F64, 3, Vec3d>, public double3
{
public:
    FW_CUDA_FUNC                    Vec3d       (void)                      { setZero(); }
    FW_CUDA_FUNC                    Vec3d       (F64 a)                     { set(a); }
    FW_CUDA_FUNC                    Vec3d       (F64 xx, F64 yy, F64 zz)    { x = xx; y = yy; z = zz; }
    FW_CUDA_FUNC                    Vec3d       (const Vec2d& xy, F64 zz)   { x = xy.x; y = xy.y; z = zz; }
    FW_CUDA_FUNC                    Vec3d       (const Vec3i& v)            { x = (F64)v.x; y = (F64)v.y; z = (F64)v.z; }
    FW_CUDA_FUNC                    Vec3d       (const Vec3f& v)            { x = v.x; y = v.y; z = v.z; }

    FW_CUDA_FUNC    const F64*      getPtr      (void) const                { return &x; }
    FW_CUDA_FUNC    F64*            getPtr      (void)                      { return &x; }
    static FW_CUDA_FUNC Vec3d       fromPtr     (const F64* ptr)            { return Vec3d(ptr[0], ptr[1], ptr[2]); }

    FW_CUDA_FUNC    operator        Vec3i       (void) const                { return Vec3i((S32)x, (S32)y, (S32)z); }
    FW_CUDA_FUNC    operator        Vec3f       (void) const                { return Vec3f((F32)x, (F32)y, (F32)z); }
    FW_CUDA_FUNC    Vec2d           getXY       (void) const                { return Vec2d(x, y); }

    FW_CUDA_FUNC    Vec3d           cross       (const Vec3d& v) const      { return Vec3d(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x); }

    template <class V> FW_CUDA_FUNC Vec3d(const VectorBase<F64, 3, V>& v) { set(v); }
    template <class V> FW_CUDA_FUNC Vec3d& operator=(const VectorBase<F64, 3, V>& v) { set(v); return *this; }
};

//------------------------------------------------------------------------

class Vec4d : public VectorBase<F64, 4, Vec4d>, public double4
{
public:
    FW_CUDA_FUNC                    Vec4d       (void)                      { setZero(); }
    FW_CUDA_FUNC                    Vec4d       (F64 a)                     { set(a); }
    FW_CUDA_FUNC                    Vec4d       (F64 xx, F64 yy, F64 zz, F64 ww) { x = xx; y = yy; z = zz; w = ww; }
    FW_CUDA_FUNC                    Vec4d       (const Vec2d& xy, F64 zz, F64 ww) { x = xy.x; y = xy.y; z = zz; w = ww; }
    FW_CUDA_FUNC                    Vec4d       (const Vec3d& xyz, F64 ww)  { x = xyz.x; y = xyz.y; z = xyz.z; w = ww; }
    FW_CUDA_FUNC                    Vec4d       (const Vec2d& xy, const Vec2d& zw) { x = xy.x; y = xy.y; z = zw.x; w = zw.y; }
    FW_CUDA_FUNC                    Vec4d       (const Vec4i& v)            { x = (F64)v.x; y = (F64)v.y; z = (F64)v.z; w = (F64)v.w; }
    FW_CUDA_FUNC                    Vec4d       (const Vec4f& v)            { x = v.x; y = v.y; z = v.z; w = v.w; }

    FW_CUDA_FUNC    const F64*      getPtr      (void) const                { return &x; }
    FW_CUDA_FUNC    F64*            getPtr      (void)                      { return &x; }
    static FW_CUDA_FUNC Vec4d       fromPtr     (const F64* ptr)            { return Vec4d(ptr[0], ptr[1], ptr[2], ptr[3]); }

    FW_CUDA_FUNC    operator        Vec4i       (void) const                { return Vec4i((S32)x, (S32)y, (S32)z, (S32)w); }
    FW_CUDA_FUNC    operator        Vec4f       (void) const                { return Vec4f((F32)x, (F32)y, (F32)z, (F32)w); }
    FW_CUDA_FUNC    Vec2d           getXY       (void) const                { return Vec2d(x, y); }
    FW_CUDA_FUNC    Vec3d           getXYZ      (void) const                { return Vec3d(x, y, z); }
    FW_CUDA_FUNC    Vec3d           getXYW      (void) const                { return Vec3d(x, y, w); }

    template <class V> FW_CUDA_FUNC Vec4d(const VectorBase<F64, 4, V>& v) { set(v); }
    template <class V> FW_CUDA_FUNC Vec4d& operator=(const VectorBase<F64, 4, V>& v) { set(v); return *this; }
};

//------------------------------------------------------------------------

template <class T, int L, class S> FW_CUDA_FUNC T lenSqr    (const VectorBase<T, L, S>& v)                  { return v.lenSqr(); }
template <class T, int L, class S> FW_CUDA_FUNC T length    (const VectorBase<T, L, S>& v)                  { return v.length(); }
template <class T, int L, class S> FW_CUDA_FUNC S normalize (const VectorBase<T, L, S>& v, T len = (T)1)    { return v.normalized(len); }
template <class T, int L, class S> FW_CUDA_FUNC T min       (const VectorBase<T, L, S>& v)                  { return v.min(); }
template <class T, int L, class S> FW_CUDA_FUNC T max       (const VectorBase<T, L, S>& v)                  { return v.max(); }
template <class T, int L, class S> FW_CUDA_FUNC T sum       (const VectorBase<T, L, S>& v)                  { return v.sum(); }
template <class T, int L, class S> FW_CUDA_FUNC S abs       (const VectorBase<T, L, S>& v)                  { return v.abs(); }

template <class T, int L, class S> FW_CUDA_FUNC S operator+     (const T& a, const VectorBase<T, L, S>& b)  { return b + a; }
template <class T, int L, class S> FW_CUDA_FUNC S operator-     (const T& a, const VectorBase<T, L, S>& b)  { return -b + a; }
template <class T, int L, class S> FW_CUDA_FUNC S operator*     (const T& a, const VectorBase<T, L, S>& b)  { return b * a; }
template <class T, int L, class S> FW_CUDA_FUNC S operator/     (const T& a, const VectorBase<T, L, S>& b)  { const T* bp = b.getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = a / bp[i]; return r; }
template <class T, int L, class S> FW_CUDA_FUNC S operator%     (const T& a, const VectorBase<T, L, S>& b)  { const T* bp = b.getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = a % bp[i]; return r; }
template <class T, int L, class S> FW_CUDA_FUNC S operator&     (const T& a, const VectorBase<T, L, S>& b)  { return b & a; }
template <class T, int L, class S> FW_CUDA_FUNC S operator|     (const T& a, const VectorBase<T, L, S>& b)  { return b | a; }
template <class T, int L, class S> FW_CUDA_FUNC S operator^     (const T& a, const VectorBase<T, L, S>& b)  { return b ^ a; }
template <class T, int L, class S> FW_CUDA_FUNC S operator<<    (const T& a, const VectorBase<T, L, S>& b)  { const T* bp = b.getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = a << bp[i]; return r; }
template <class T, int L, class S> FW_CUDA_FUNC S operator>>    (const T& a, const VectorBase<T, L, S>& b)  { const T* bp = b.getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = a >> bp[i]; return r; }

template <class T, int L, class S, class V> FW_CUDA_FUNC T dot(const VectorBase<T, L, S>& a, const VectorBase<T, L, V>& b) { return a.dot(b); }

FW_CUDA_FUNC Vec2f  perpendicular   (const Vec2f& v)                    { return v.perpendicular(); }
FW_CUDA_FUNC Vec2d  perpendicular   (const Vec2d& v)                    { return v.perpendicular(); }
FW_CUDA_FUNC F32    cross           (const Vec2f& a, const Vec2f& b)    { return a.cross(b); }
FW_CUDA_FUNC F64    cross           (const Vec2d& a, const Vec2d& b)    { return a.cross(b); }
FW_CUDA_FUNC Vec3f  cross           (const Vec3f& a, const Vec3f& b)    { return a.cross(b); }
FW_CUDA_FUNC Vec3d  cross           (const Vec3d& a, const Vec3d& b)    { return a.cross(b); }

#define MINMAX(T) \
    FW_CUDA_FUNC T min(const T& a, const T& b)                          { return a.min(b); } \
    FW_CUDA_FUNC T min(T& a, T& b)                                      { return a.min(b); } \
    FW_CUDA_FUNC T max(const T& a, const T& b)                          { return a.max(b); } \
    FW_CUDA_FUNC T max(T& a, T& b)                                      { return a.max(b); } \
    FW_CUDA_FUNC T min(const T& a, const T& b, const T& c)              { return a.min(b).min(c); } \
    FW_CUDA_FUNC T min(T& a, T& b, T& c)                                { return a.min(b).min(c); } \
    FW_CUDA_FUNC T max(const T& a, const T& b, const T& c)              { return a.max(b).max(c); } \
    FW_CUDA_FUNC T max(T& a, T& b, T& c)                                { return a.max(b).max(c); } \
    FW_CUDA_FUNC T min(const T& a, const T& b, const T& c, const T& d)  { return a.min(b).min(c).min(d); } \
    FW_CUDA_FUNC T min(T& a, T& b, T& c, T& d)                          { return a.min(b).min(c).min(d); } \
    FW_CUDA_FUNC T max(const T& a, const T& b, const T& c, const T& d)  { return a.max(b).max(c).max(d); } \
    FW_CUDA_FUNC T max(T& a, T& b, T& c, T& d)                          { return a.max(b).max(c).max(d); } \
    FW_CUDA_FUNC T clamp(const T& v, const T& lo, const T& hi)          { return v.clamp(lo, hi); } \
    FW_CUDA_FUNC T clamp(T& v, T& lo, T& hi)                            { return v.clamp(lo, hi); }

MINMAX(Vec2i) MINMAX(Vec3i) MINMAX(Vec4i)
MINMAX(Vec2f) MINMAX(Vec3f) MINMAX(Vec4f)
MINMAX(Vec2d) MINMAX(Vec3d) MINMAX(Vec4d)
#undef MINMAX

//------------------------------------------------------------------------

template <class T, int L, class S> class MatrixBase
{
public:
    FW_CUDA_FUNC                    MatrixBase  (void)                      {}

    template <class V> static FW_CUDA_FUNC S    translate   (const VectorBase<T, L - 1, V>& v);
    template <class V> static FW_CUDA_FUNC S    scale       (const VectorBase<T, L - 1, V>& v);
    template <class V> static FW_CUDA_FUNC S    scale       (const VectorBase<T, L, V>& v);

    FW_CUDA_FUNC    const T*        getPtr      (void) const                { return ((S*)this)->getPtr(); }
    FW_CUDA_FUNC    T*              getPtr      (void)                      { return ((S*)this)->getPtr(); }
    FW_CUDA_FUNC    const T&        get         (int idx) const             { FW_ASSERT(idx >= 0 && idx < L * L); return getPtr()[idx]; }
    FW_CUDA_FUNC    T&              get         (int idx)                   { FW_ASSERT(idx >= 0 && idx < L * L); return getPtr()[idx]; }
    FW_CUDA_FUNC    const T&        get         (int r, int c) const        { FW_ASSERT(r >= 0 && r < L && c >= 0 && c < L); return getPtr()[r + c * L]; }
    FW_CUDA_FUNC    T&              get         (int r, int c)              { FW_ASSERT(r >= 0 && r < L && c >= 0 && c < L); return getPtr()[r + c * L]; }
    FW_CUDA_FUNC    T               set         (int idx, const T& a)       { T& slot = get(idx); T old = slot; slot = a; return old; }
    FW_CUDA_FUNC    T               set         (int r, int c, const T& a)  { T& slot = get(r, c); T old = slot; slot = a; return old; }
    FW_CUDA_FUNC    const Vector<T, L>& col     (int c) const               { FW_ASSERT(c >= 0 && c < L); return *(const Vector<T, L>*)(getPtr() + c * L); }
    FW_CUDA_FUNC    Vector<T, L>&   col         (int c)                     { FW_ASSERT(c >= 0 && c < L); return *(Vector<T, L>*)(getPtr() + c * L); }
    FW_CUDA_FUNC    const Vector<T, L>& getCol  (int c) const               { return col(c); }
    FW_CUDA_FUNC    Vector<T, L>    getRow      (int r) const;

    FW_CUDA_FUNC    void            set         (const T& a)                { for (int i = 0; i < L * L; i++) get(i) = a; }
    FW_CUDA_FUNC    void            set         (const T* ptr)              { FW_ASSERT(ptr); for (int i = 0; i < L * L; i++) get(i) = ptr[i]; }
    FW_CUDA_FUNC    void            setZero     (void)                      { set((T)0); }
    FW_CUDA_FUNC    void            setIdentity (void)                      { setZero(); for (int i = 0; i < L; i++) get(i, i) = (T)1; }

#if !FW_CUDA
                    void            print       (void) const;
#endif

    FW_CUDA_FUNC    T               det         (void) const;
    FW_CUDA_FUNC    S               transposed  (void) const;
    FW_CUDA_FUNC    S               inverted    (void) const;
    FW_CUDA_FUNC    void            transpose   (void)                      { set(transposed()); }
    FW_CUDA_FUNC    void            invert      (void)                      { set(inverted()); }

    FW_CUDA_FUNC    const T&        operator()  (int r, int c) const        { return get(r, c); }
    FW_CUDA_FUNC    T&              operator()  (int r, int c)              { return get(r, c); }

    FW_CUDA_FUNC    S&              operator=   (const T& a)                { set(a); return *(S*)this; }
    FW_CUDA_FUNC    S&              operator+=  (const T& a)                { set(operator+(a)); return *(S*)this; }
    FW_CUDA_FUNC    S&              operator-=  (const T& a)                { set(operator-(a)); return *(S*)this; }
    FW_CUDA_FUNC    S&              operator*=  (const T& a)                { set(operator*(a)); return *(S*)this; }
    FW_CUDA_FUNC    S&              operator/=  (const T& a)                { set(operator/(a)); return *(S*)this; }
    FW_CUDA_FUNC    S&              operator%=  (const T& a)                { set(operator%(a)); return *(S*)this; }
    FW_CUDA_FUNC    S&              operator&=  (const T& a)                { set(operator&(a)); return *(S*)this; }
    FW_CUDA_FUNC    S&              operator|=  (const T& a)                { set(operator|(a)); return *(S*)this; }
    FW_CUDA_FUNC    S&              operator^=  (const T& a)                { set(operator^(a)); return *(S*)this; }
    FW_CUDA_FUNC    S&              operator<<= (const T& a)                { set(operator<<(a)); return *(S*)this; }
    FW_CUDA_FUNC    S&              operator>>= (const T& a)                { set(operator>>(a)); return *(S*)this; }

    FW_CUDA_FUNC    S               operator+   (void) const                { return *this; }
    FW_CUDA_FUNC    S               operator-   (void) const                { S r; for (int i = 0; i < L * L; i++) r.get(i) = -get(i); return r; }
    FW_CUDA_FUNC    S               operator~   (void) const                { S r; for (int i = 0; i < L * L; i++) r.get(i) = ~get(i); return r; }

    FW_CUDA_FUNC    S               operator+   (const T& a) const          { S r; for (int i = 0; i < L * L; i++) r.get(i) = get(i) + a; return r; }
    FW_CUDA_FUNC    S               operator-   (const T& a) const          { S r; for (int i = 0; i < L * L; i++) r.get(i) = get(i) - a; return r; }
    FW_CUDA_FUNC    S               operator*   (const T& a) const          { S r; for (int i = 0; i < L * L; i++) r.get(i) = get(i) * a; return r; }
    FW_CUDA_FUNC    S               operator/   (const T& a) const          { S r; for (int i = 0; i < L * L; i++) r.get(i) = get(i) / a; return r; }
    FW_CUDA_FUNC    S               operator%   (const T& a) const          { S r; for (int i = 0; i < L * L; i++) r.get(i) = get(i) % a; return r; }
    FW_CUDA_FUNC    S               operator&   (const T& a) const          { S r; for (int i = 0; i < L * L; i++) r.get(i) = get(i) & a; return r; }
    FW_CUDA_FUNC    S               operator|   (const T& a) const          { S r; for (int i = 0; i < L * L; i++) r.get(i) = get(i) | a; return r; }
    FW_CUDA_FUNC    S               operator^   (const T& a) const          { S r; for (int i = 0; i < L * L; i++) r.get(i) = get(i) ^ a; return r; }
    FW_CUDA_FUNC    S               operator<<  (const T& a) const          { S r; for (int i = 0; i < L * L; i++) r.get(i) = get(i) << a; return r; }
    FW_CUDA_FUNC    S               operator>>  (const T& a) const          { S r; for (int i = 0; i < L * L; i++) r.get(i) = get(i) >> a; return r; }

    template <class V> FW_CUDA_FUNC void    setCol      (int c, const VectorBase<T, L, V>& v)   { col(c) = v; }
    template <class V> FW_CUDA_FUNC void    setRow      (int r, const VectorBase<T, L, V>& v);
    template <class V> FW_CUDA_FUNC void    set         (const MatrixBase<T, L, V>& v)          { set(v.getPtr()); }

    template <class V> FW_CUDA_FUNC S&      operator=   (const MatrixBase<T, L, V>& v)          { set(v); return *(S*)this; }
    template <class V> FW_CUDA_FUNC S&      operator+=  (const MatrixBase<T, L, V>& v)          { set(operator+(v)); return *(S*)this; }
    template <class V> FW_CUDA_FUNC S&      operator-=  (const MatrixBase<T, L, V>& v)          { set(operator-(v)); return *(S*)this; }
    template <class V> FW_CUDA_FUNC S&      operator*=  (const MatrixBase<T, L, V>& v)          { set(operator*(v)); return *(S*)this; }
    template <class V> FW_CUDA_FUNC S&      operator/=  (const MatrixBase<T, L, V>& v)          { set(operator/(v)); return *(S*)this; }
    template <class V> FW_CUDA_FUNC S&      operator%=  (const MatrixBase<T, L, V>& v)          { set(operator%(v)); return *(S*)this; }
    template <class V> FW_CUDA_FUNC S&      operator&=  (const MatrixBase<T, L, V>& v)          { set(operator&(v)); return *(S*)this; }
    template <class V> FW_CUDA_FUNC S&      operator|=  (const MatrixBase<T, L, V>& v)          { set(operator|(v)); return *(S*)this; }
    template <class V> FW_CUDA_FUNC S&      operator^=  (const MatrixBase<T, L, V>& v)          { set(operator^(v)); return *(S*)this; }
    template <class V> FW_CUDA_FUNC S&      operator<<= (const MatrixBase<T, L, V>& v)          { set(operator<<(v)); return *(S*)this; }
    template <class V> FW_CUDA_FUNC S&      operator>>= (const MatrixBase<T, L, V>& v)          { set(operator>>(v)); return *(S*)this; }

    template <class V> FW_CUDA_FUNC V       operator*   (const VectorBase<T, L, V>& v) const;
    template <class V> FW_CUDA_FUNC V       operator*   (const VectorBase<T, L - 1, V>& v) const;

    template <class V> FW_CUDA_FUNC S       operator+   (const MatrixBase<T, L, V>& v) const    { S r; for (int i = 0; i < L * L; i++) r.get(i) = get(i) + v.get(i); return r; }
    template <class V> FW_CUDA_FUNC S       operator-   (const MatrixBase<T, L, V>& v) const    { S r; for (int i = 0; i < L * L; i++) r.get(i) = get(i) - v.get(i); return r; }
    template <class V> FW_CUDA_FUNC S       operator*   (const MatrixBase<T, L, V>& v) const;
    template <class V> FW_CUDA_FUNC S       operator/   (const MatrixBase<T, L, V>& v) const    { return operator*(v.inverted()); }
    template <class V> FW_CUDA_FUNC S       operator%   (const MatrixBase<T, L, V>& v) const    { S r; for (int i = 0; i < L * L; i++) r.get(i) = get(i) % v.get(i); return r; }
    template <class V> FW_CUDA_FUNC S       operator&   (const MatrixBase<T, L, V>& v) const    { S r; for (int i = 0; i < L * L; i++) r.get(i) = get(i) & v.get(i); return r; }
    template <class V> FW_CUDA_FUNC S       operator|   (const MatrixBase<T, L, V>& v) const    { S r; for (int i = 0; i < L * L; i++) r.get(i) = get(i) | v.get(i); return r; }
    template <class V> FW_CUDA_FUNC S       operator^   (const MatrixBase<T, L, V>& v) const    { S r; for (int i = 0; i < L * L; i++) r.get(i) = get(i) ^ v.get(i); return r; }
    template <class V> FW_CUDA_FUNC S       operator<<  (const MatrixBase<T, L, V>& v) const    { S r; for (int i = 0; i < L * L; i++) r.get(i) = get(i) << v.get(i); return r; }
    template <class V> FW_CUDA_FUNC S       operator>>  (const MatrixBase<T, L, V>& v) const    { S r; for (int i = 0; i < L * L; i++) r.get(i) = get(i) >> v.get(i); return r; }

    template <class V> FW_CUDA_FUNC bool    operator==  (const MatrixBase<T, L, V>& v) const    { for (int i = 0; i < L * L; i++) if (get(i) != v.get(i)) return false; return true; }
    template <class V> FW_CUDA_FUNC bool    operator!=  (const MatrixBase<T, L, V>& v) const    { return (!operator==(v)); }
};

//------------------------------------------------------------------------

template <class T, int L> class Matrix : public MatrixBase<T, L, Matrix<T, L> >
{
public:
    FW_CUDA_FUNC                    Matrix      (void)                      { setIdentity(); }
    FW_CUDA_FUNC    explicit        Matrix      (T a)                       { set(a); }

    FW_CUDA_FUNC    const T*        getPtr      (void) const                { return m_values; }
    FW_CUDA_FUNC    T*              getPtr      (void)                      { return m_values; }
    static FW_CUDA_FUNC Matrix      fromPtr     (const T* ptr)              { Matrix v; v.set(ptr); return v; }

    template <class V> FW_CUDA_FUNC Matrix(const MatrixBase<T, L, V>& v) { set(v); }
    template <class V> FW_CUDA_FUNC Matrix& operator=(const MatrixBase<T, L, V>& v) { set(v); return *this; }

private:
    T               m_values[L * L];
};

//------------------------------------------------------------------------

class Mat2f : public MatrixBase<F32, 2, Mat2f>
{
public:
    FW_CUDA_FUNC                    Mat2f       (void)                      { setIdentity(); }
    FW_CUDA_FUNC    explicit        Mat2f       (F32 a)                     { set(a); }

    FW_CUDA_FUNC    const F32*      getPtr      (void) const                { return &m00; }
    FW_CUDA_FUNC    F32*            getPtr      (void)                      { return &m00; }
    static FW_CUDA_FUNC Mat2f       fromPtr     (const F32* ptr)            { Mat2f v; v.set(ptr); return v; }

    template <class V> FW_CUDA_FUNC Mat2f(const MatrixBase<F32, 2, V>& v) { set(v); }
    template <class V> FW_CUDA_FUNC Mat2f& operator=(const MatrixBase<F32, 2, V>& v) { set(v); return *this; }

public:
    F32             m00, m10;
    F32             m01, m11;
};

//------------------------------------------------------------------------

class Mat3f : public MatrixBase<F32, 3, Mat3f>
{
public:
    FW_CUDA_FUNC                    Mat3f       (void)                      { setIdentity(); }
    FW_CUDA_FUNC    explicit        Mat3f       (F32 a)                     { set(a); }

    FW_CUDA_FUNC    const F32*      getPtr      (void) const                { return &m00; }
    FW_CUDA_FUNC    F32*            getPtr      (void)                      { return &m00; }
    static FW_CUDA_FUNC Mat3f       fromPtr     (const F32* ptr)            { Mat3f v; v.set(ptr); return v; }

    template <class V> FW_CUDA_FUNC Mat3f(const MatrixBase<F32, 3, V>& v) { set(v); }
    template <class V> FW_CUDA_FUNC Mat3f& operator=(const MatrixBase<F32, 3, V>& v) { set(v); return *this; }

#if !FW_CUDA 
	static			Mat3f			rotation	(const Vec3f& axis, F32 angle);		// Rotation of "angle" radians around "axis". Axis must be unit!
#endif

public:
    F32             m00, m10, m20;
    F32             m01, m11, m21;
    F32             m02, m12, m22;
};

//------------------------------------------------------------------------

class Mat4f : public MatrixBase<F32, 4, Mat4f>
{
public:
    FW_CUDA_FUNC                    Mat4f       (void)                      { setIdentity(); }
    FW_CUDA_FUNC    explicit        Mat4f       (F32 a)                     { set(a); }

    FW_CUDA_FUNC    const F32*      getPtr      (void) const                { return &m00; }
    FW_CUDA_FUNC    F32*            getPtr      (void)                      { return &m00; }
    static FW_CUDA_FUNC Mat4f       fromPtr     (const F32* ptr)            { Mat4f v; v.set(ptr); return v; }

#if !FW_CUDA
    Mat3f                           getXYZ      (void) const;
    static Mat4f                    fitToView   (const Vec2f& pos, const Vec2f& size, const Vec2f& viewSize);
    static Mat4f                    perspective (F32 fov, F32 nearDist, F32 farDist);
#endif

    template <class V> FW_CUDA_FUNC Mat4f(const MatrixBase<F32, 4, V>& v) { set(v); }
    template <class V> FW_CUDA_FUNC Mat4f& operator=(const MatrixBase<F32, 4, V>& v) { set(v); return *this; }

public:
    F32             m00, m10, m20, m30;
    F32             m01, m11, m21, m31;
    F32             m02, m12, m22, m32;
    F32             m03, m13, m23, m33;
};

//------------------------------------------------------------------------

class Mat2d : public MatrixBase<F64, 2, Mat2d>
{
public:
    FW_CUDA_FUNC                    Mat2d       (void)                      { setIdentity(); }
    FW_CUDA_FUNC                    Mat2d       (const Mat2f& a)            { for (int i = 0; i < 2 * 2; i++) set(i, (F64)a.get(i)); }
    FW_CUDA_FUNC    explicit        Mat2d       (F64 a)                     { set(a); }

    FW_CUDA_FUNC    const F64*      getPtr      (void) const                { return &m00; }
    FW_CUDA_FUNC    F64*            getPtr      (void)                      { return &m00; }
    static FW_CUDA_FUNC Mat2d       fromPtr     (const F64* ptr)            { Mat2d v; v.set(ptr); return v; }

    FW_CUDA_FUNC    operator        Mat2f       (void) const                { Mat2f r; for (int i = 0; i < 2 * 2; i++) r.set(i, (F32)get(i)); return r; }

    template <class V> FW_CUDA_FUNC Mat2d(const MatrixBase<F64, 2, V>& v) { set(v); }
    template <class V> FW_CUDA_FUNC Mat2d& operator=(const MatrixBase<F64, 2, V>& v) { set(v); return *this; }

public:
    F64             m00, m10;
    F64             m01, m11;
};

//------------------------------------------------------------------------

class Mat3d : public MatrixBase<F64, 3, Mat3d>
{
public:
    FW_CUDA_FUNC                    Mat3d       (void)                      { setIdentity(); }
    FW_CUDA_FUNC                    Mat3d       (const Mat3f& a)            { for (int i = 0; i < 3 * 3; i++) set(i, (F64)a.get(i)); }
    FW_CUDA_FUNC    explicit        Mat3d       (F64 a)                     { set(a); }

    FW_CUDA_FUNC    const F64*      getPtr      (void) const                { return &m00; }
    FW_CUDA_FUNC    F64*            getPtr      (void)                      { return &m00; }
    static FW_CUDA_FUNC Mat3d       fromPtr     (const F64* ptr)            { Mat3d v; v.set(ptr); return v; }

    FW_CUDA_FUNC    operator        Mat3f       (void) const                { Mat3f r; for (int i = 0; i < 3 * 3; i++) r.set(i, (F32)get(i)); return r; }

#if !FW_CUDA 
	static			Mat3d			rotation	(const Vec3d& axis, F64 angle);		// Rotation of "angle" radians around "axis". Axis must be unit!
#endif

    template <class V> FW_CUDA_FUNC Mat3d(const MatrixBase<F64, 3, V>& v) { set(v); }
    template <class V> FW_CUDA_FUNC Mat3d& operator=(const MatrixBase<F64, 3, V>& v) { set(v); return *this; }

public:
    F64             m00, m10, m20;
    F64             m01, m11, m21;
    F64             m02, m12, m22;
};

//------------------------------------------------------------------------

class Mat4d : public MatrixBase<F64, 4, Mat4d>
{
public:
    FW_CUDA_FUNC                    Mat4d       (void)                      { setIdentity(); }
    FW_CUDA_FUNC                    Mat4d       (const Mat4f& a)            { for (int i = 0; i < 4 * 4; i++) set(i, (F64)a.get(i)); }
    FW_CUDA_FUNC    explicit        Mat4d       (F64 a)                     { set(a); }

    FW_CUDA_FUNC    const F64*      getPtr      (void) const                { return &m00; }
    FW_CUDA_FUNC    F64*            getPtr      (void)                      { return &m00; }
    static FW_CUDA_FUNC Mat4d       fromPtr     (const F64* ptr)            { Mat4d v; v.set(ptr); return v; }

    FW_CUDA_FUNC    operator        Mat4f       (void) const                { Mat4f r; for (int i = 0; i < 4 * 4; i++) r.set(i, (F32)get(i)); return r; }

    template <class V> FW_CUDA_FUNC Mat4d(const MatrixBase<F64, 4, V>& v) { set(v); }
    template <class V> FW_CUDA_FUNC Mat4d& operator=(const MatrixBase<F64, 4, V>& v) { set(v); return *this; }

public:
    F64             m00, m10, m20, m30;
    F64             m01, m11, m21, m31;
    F64             m02, m12, m22, m32;
    F64             m03, m13, m23, m33;
};

//------------------------------------------------------------------------

template <class T, int L, class S> FW_CUDA_FUNC Matrix<T, L> outerProduct(const VectorBase<T, L, S>& a, const VectorBase<T, L, S>& b);

template <class T, int L, class S> FW_CUDA_FUNC T det           (const MatrixBase<T, L, S>& v)  { return v.det(); }
template <class T, int L, class S> FW_CUDA_FUNC S transpose     (const MatrixBase<T, L, S>& v)  { return v.transposed(); }
template <class T, int L, class S> FW_CUDA_FUNC S invert        (const MatrixBase<T, L, S>& v)  { return v.inverted(); }

template <class T, int L, class S> FW_CUDA_FUNC S operator+     (const T& a, const MatrixBase<T, L, S>& b)  { return b + a; }
template <class T, int L, class S> FW_CUDA_FUNC S operator-     (const T& a, const MatrixBase<T, L, S>& b)  { return -b + a; }
template <class T, int L, class S> FW_CUDA_FUNC S operator*     (const T& a, const MatrixBase<T, L, S>& b)  { return b * a; }
template <class T, int L, class S> FW_CUDA_FUNC S operator/     (const T& a, const MatrixBase<T, L, S>& b)  { S r; for (int i = 0; i < L * L; i++) r.get(i) = a / b.get(i); return r; }
template <class T, int L, class S> FW_CUDA_FUNC S operator%     (const T& a, const MatrixBase<T, L, S>& b)  { S r; for (int i = 0; i < L * L; i++) r.get(i) = a % b.get(i); return r; }
template <class T, int L, class S> FW_CUDA_FUNC S operator&     (const T& a, const MatrixBase<T, L, S>& b)  { return b & a; }
template <class T, int L, class S> FW_CUDA_FUNC S operator|     (const T& a, const MatrixBase<T, L, S>& b)  { return b | a; }
template <class T, int L, class S> FW_CUDA_FUNC S operator^     (const T& a, const MatrixBase<T, L, S>& b)  { return b ^ a; }
template <class T, int L, class S> FW_CUDA_FUNC S operator<<    (const T& a, const MatrixBase<T, L, S>& b)  { S r; for (int i = 0; i < L * L; i++) r.get(i) = a << b.get(i); return r; }
template <class T, int L, class S> FW_CUDA_FUNC S operator>>    (const T& a, const MatrixBase<T, L, S>& b)  { S r; for (int i = 0; i < L * L; i++) r.get(i) = a >> b.get(i); return r; }

//------------------------------------------------------------------------

FW_CUDA_CONST int c_popc8LUT[] =
{
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8,
};

FW_CUDA_FUNC int popc8(U32 mask)
{
    return c_popc8LUT[mask & 0xFFu];
}

FW_CUDA_FUNC int popc16(U32 mask)
{
    return c_popc8LUT[mask & 0xFFu] + c_popc8LUT[(mask >> 8) & 0xFFu];
}

FW_CUDA_FUNC int popc32(U32 mask)
{
    int result = c_popc8LUT[mask & 0xFFu];
    result += c_popc8LUT[(mask >> 8) & 0xFFu];
    result += c_popc8LUT[(mask >> 16) & 0xFFu];
    result += c_popc8LUT[mask >> 24];
    return result;
}

FW_CUDA_FUNC int popc64(U64 mask)
{
    U32 lo = (U32)mask;
    U32 hi = (U32)(mask >> 32);
    int result = c_popc8LUT[lo & 0xffu] + c_popc8LUT[hi & 0xffu];
    result += c_popc8LUT[(lo >> 8) & 0xffu] + c_popc8LUT[(hi >> 8) & 0xffu];
    result += c_popc8LUT[(lo >> 16) & 0xffu] + c_popc8LUT[(hi >> 16) & 0xffu];
    result += c_popc8LUT[lo >> 24] + c_popc8LUT[hi >> 24];
    return result;
}

//------------------------------------------------------------------------

template <class T, int L, class S> template <class V> S MatrixBase<T, L, S>::translate(const VectorBase<T, L - 1, V>& v)
{
    S r;
    for (int i = 0; i < L - 1; i++)
        r(i, L - 1) = v[i];
    return r;
}

//------------------------------------------------------------------------

template <class T, int L, class S> template <class V> S MatrixBase<T, L, S>::scale(const VectorBase<T, L - 1, V>& v)
{
    S r;
    for (int i = 0; i < L - 1; i++)
        r(i, i) = v[i];
    return r;
}

//------------------------------------------------------------------------

template <class T, int L, class S> template <class V> S MatrixBase<T, L, S>::scale(const VectorBase<T, L, V>& v)
{
    S r;
    for (int i = 0; i < L; i++)
        r(i, i) = v[i];
    return r;
}

//------------------------------------------------------------------------

template <class T, int L, class S> Vector<T, L> MatrixBase<T, L, S>::getRow(int idx) const
{
    Vector<T, L> r;
    for (int i = 0; i < L; i++)
        r[i] = get(idx, i);
    return r;
}

//------------------------------------------------------------------------

#if !FW_CUDA
template <class T, int L, class S> void MatrixBase<T, L, S>::print(void) const
{
    for (int i = 0; i < L; i++)
    {
        for (int j = 0; j < L; j++)
            printf("%-16g", (F64)get(i, j));
        printf("\n");
    }
}
#endif

//------------------------------------------------------------------------

template <class T, int L, class S> FW_CUDA_FUNC T detImpl(const MatrixBase<T, L, S>& v)
{
    T r = (T)0;
    T s = (T)1;
    for (int i = 0; i < L; i++)
    {
        Matrix<T, L - 1> sub;
        for (int j = 0; j < L - 1; j++)
            for (int k = 0; k < L - 1; k++)
                sub(j, k) = v((j < i) ? j : j + 1, k + 1);
        r += sub.det() * v(i, 0) * s;
        s = -s;
    }
    return r;
}

//------------------------------------------------------------------------

template <class T, class S> FW_CUDA_FUNC T detImpl(const MatrixBase<T, 1, S>& v)
{
    return v(0, 0);
}

//------------------------------------------------------------------------

template <class T, class S> FW_CUDA_FUNC T detImpl(const MatrixBase<T, 2, S>& v)
{
    return v(0, 0) * v(1, 1) - v(0, 1) * v(1, 0);
}

//------------------------------------------------------------------------

template <class T, class S> FW_CUDA_FUNC T detImpl(const MatrixBase<T, 3, S>& v)
{
    return v(0, 0) * v(1, 1) * v(2, 2) - v(0, 0) * v(1, 2) * v(2, 1) +
           v(1, 0) * v(2, 1) * v(0, 2) - v(1, 0) * v(2, 2) * v(0, 1) +
           v(2, 0) * v(0, 1) * v(1, 2) - v(2, 0) * v(0, 2) * v(1, 1);
}

//------------------------------------------------------------------------

template <class T, int L, class S> T MatrixBase<T, L, S>::det(void) const
{
    return detImpl(*this);
}

//------------------------------------------------------------------------

template <class T, int L, class S> S MatrixBase<T, L, S>::transposed(void) const
{
    S r;
    for (int i = 0; i < L; i++)
        for (int j = 0; j < L; j++)
            r(i, j) = get(j, i);
    return r;
}

//------------------------------------------------------------------------

template <class T, int L, class S> S MatrixBase<T, L, S>::inverted(void) const
{
    S r;
    T d = (T)0;
    T si = (T)1;
    for (int i = 0; i < L; i++)
    {
        T sj = si;
        for (int j = 0; j < L; j++)
        {
            Matrix<T, L - 1> sub;
            for (int k = 0; k < L - 1; k++)
                for (int l = 0; l < L - 1; l++)
                    sub(k, l) = get((k < j) ? k : k + 1, (l < i) ? l : l + 1);
            T dd = sub.det() * sj;
            r(i, j) = dd;
            d += dd * get(j, i);
            sj = -sj;
        }
        si = -si;
    }
    return r * rcp(d) * L;
}

//------------------------------------------------------------------------

template <class T, int L, class S> template <class V> void MatrixBase<T, L, S>::setRow(int idx, const VectorBase<T, L, V>& v)
{
    for (int i = 0; i < L; i++)
        get(idx, i) = v[i];
}

//------------------------------------------------------------------------

template <class T, int L, class S> template<class V> V MatrixBase<T, L, S>::operator*(const VectorBase<T, L, V>& v) const
{
    V r;
    for (int i = 0; i < L; i++)
    {
        T rr = (T)0;
        for (int j = 0; j < L; j++)
            rr += get(i, j) * v[j];
        r[i] = rr;
    }
    return r;
}

//------------------------------------------------------------------------

template <class T, int L, class S> template<class V> V MatrixBase<T, L, S>::operator*(const VectorBase<T, L - 1, V>& v) const
{
    T w = get(L - 1, L - 1);
    for (int i = 0; i < L - 1; i++)
        w += get(L - 1, i) * v[i];
    w = rcp(w);

    V r;
    for (int i = 0; i < L - 1; i++)
    {
        T rr = get(i, L - 1);
        for (int j = 0; j < L - 1; j++)
            rr += get(i, j) * v[j];
        r[i] = rr * w;
    }
    return r;
}

//------------------------------------------------------------------------

template <class T, int L, class S> template <class V> S MatrixBase<T, L, S>::operator*(const MatrixBase<T, L, V>& v) const
{
    S r;
    for (int i = 0; i < L; i++)
    {
        for (int j = 0; j < L; j++)
        {
            T rr = (T)0;
            for (int k = 0; k < L; k++)
                rr += get(i, k) * v(k, j);
            r(i, j) = rr;
        }
    }
    return r;
}

//------------------------------------------------------------------------

template <class T, int L, class S> Matrix<T, L> outerProduct(const VectorBase<T, L, S>& a, const VectorBase<T, L, S>& b)
{
    Matrix<T, L> res;
    for (int i = 0; i < L; i++)
        for (int j = 0; j < L; j++)
            res.get(i, j) = a.get(i) * b.get(j);
    return res;
}

//------------------------------------------------------------------------
}
