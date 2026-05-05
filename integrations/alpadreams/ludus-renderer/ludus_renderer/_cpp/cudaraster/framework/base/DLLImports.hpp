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
#include "base/Defs.hpp"

//------------------------------------------------------------------------

#define FW_USE_CUDA 1
#define FW_USE_GLEW 0

//------------------------------------------------------------------------

#if (FW_USE_CUDA)
#   include <cuda.h>
#   pragma warning(push,3)
#       include <vector_functions.h> // float4, etc.
#   pragma warning(pop)
#endif

#if (!FW_CUDA)
#   define _WIN32_WINNT 0x0600
#   define WIN32_LEAN_AND_MEAN
#   define _KERNEL32_
#   define _WINMM_
#   include <windows.h>
#   undef min
#   undef max

#   pragma warning(push,3)
#   include <mmsystem.h>
#   pragma warning(pop)

#   define _SHLWAPI_
#   include <shlwapi.h>
#endif

//------------------------------------------------------------------------

namespace FW
{
#if (!FW_CUDA)
void    setCudaDLLName      (const String& name);
void    initDLLImports      (void);
void    initGLImports       (void);
void    deinitDLLImports    (void);
#endif
}

//------------------------------------------------------------------------
// CUDA definitions.
//------------------------------------------------------------------------

#if (!FW_USE_CUDA)
#   define CUDA_VERSION 2010
#   define CUDAAPI __stdcall

typedef enum { CUDA_SUCCESS = 0}        CUresult;
typedef struct { FW::S32 x, y; }        int2;
typedef struct { FW::S32 x, y, z; }     int3;
typedef struct { FW::S32 x, y, z, w; }  int4;
typedef struct { FW::F32 x, y; }        float2;
typedef struct { FW::F32 x, y, z; }     float3;
typedef struct { FW::F32 x, y, z, w; }  float4;
typedef struct { FW::F64 x, y; }        double2;
typedef struct { FW::F64 x, y, z; }     double3;
typedef struct { FW::F64 x, y, z, w; }  double4;

typedef void*   CUfunction;
typedef void*   CUmodule;
typedef int     CUdevice;
typedef size_t  CUdeviceptr;
typedef void*   CUcontext;
typedef void*   CUdevprop;
typedef int     CUdevice_attribute;
typedef int     CUjit_option;
typedef void*   CUtexref;
typedef void*   CUarray;
typedef int     CUarray_format;
typedef int     CUaddress_mode;
typedef int     CUfilter_mode;
typedef void*   CUstream;
typedef void*   CUevent;
typedef void*   CUDA_MEMCPY2D;
typedef void*   CUDA_MEMCPY3D;
typedef void*   CUDA_ARRAY_DESCRIPTOR;
typedef void*   CUDA_ARRAY3D_DESCRIPTOR;
typedef int     CUfunction_attribute;

#endif

#if (CUDA_VERSION < 3010)
typedef void* CUsurfref;
#endif

#if (CUDA_VERSION < 3020)
typedef unsigned int    CUsize_t;
#else
typedef size_t          CUsize_t;
#endif

//------------------------------------------------------------------------
// GL definitions.
//------------------------------------------------------------------------

#if (!FW_CUDA && FW_USE_GLEW)
#   define GL_FUNC_AVAILABLE(NAME) (NAME != NULL)
#   define GLEW_STATIC
#   include "3rdparty/glew/include/GL/glew.h"
#   include "3rdparty/glew/include/GL/wglew.h"
#   if FW_USE_CUDA
#       include <cudaGL.h>
#   endif

#elif (!FW_CUDA && !FW_USE_GLEW)
#   define GL_FUNC_AVAILABLE(NAME) (isAvailable_ ## NAME())
#   include <GL/gl.h>
#   if FW_USE_CUDA
#       include <cudaGL.h>
#   endif

typedef char            GLchar;
typedef ptrdiff_t       GLintptr;
typedef ptrdiff_t       GLsizeiptr;
typedef unsigned int    GLhandleARB;

#define GL_ALPHA32F_ARB                     0x8816
#define GL_ARRAY_BUFFER                     0x8892
#define GL_BUFFER_SIZE                      0x8764
#define GL_COLOR_ATTACHMENT0                0x8CE0
#define GL_COLOR_ATTACHMENT1                0x8CE1
#define GL_COLOR_ATTACHMENT2                0x8CE2
#define GL_COMPILE_STATUS                   0x8B81
#define GL_DEPTH_ATTACHMENT                 0x8D00
#define GL_ELEMENT_ARRAY_BUFFER             0x8893
#define GL_FRAGMENT_SHADER                  0x8B30
#define GL_FRAMEBUFFER                      0x8D40
#define GL_FUNC_ADD                         0x8006
#define GL_GENERATE_MIPMAP                  0x8191
#define GL_GEOMETRY_INPUT_TYPE_ARB          0x8DDB
#define GL_GEOMETRY_OUTPUT_TYPE_ARB         0x8DDC
#define GL_GEOMETRY_SHADER_ARB              0x8DD9
#define GL_GEOMETRY_VERTICES_OUT_ARB        0x8DDA
#define GL_INFO_LOG_LENGTH                  0x8B84
#define GL_INVALID_FRAMEBUFFER_OPERATION    0x0506
#define GL_LINK_STATUS                      0x8B82
#define GL_PIXEL_PACK_BUFFER                0x88EB
#define GL_PIXEL_UNPACK_BUFFER              0x88EC
#define GL_RENDERBUFFER                     0x8D41
#define GL_RGB32F                           0x8815
#define GL_RGBA32F                          0x8814
#define GL_RGBA32UI                         0x8D70
#define GL_RGBA_INTEGER                     0x8D99
#define GL_STATIC_DRAW                      0x88E4
#define GL_DYNAMIC_COPY                     0x88EA
#define GL_TEXTURE0                         0x84C0
#define GL_TEXTURE1                         0x84C1
#define GL_TEXTURE2                         0x84C2
#define GL_TEXTURE_3D                       0x806F
#define GL_TEXTURE_CUBE_MAP                 0x8513
#define GL_TEXTURE_CUBE_MAP_POSITIVE_X      0x8515
#define GL_UNSIGNED_SHORT_5_5_5_1           0x8034
#define GL_UNSIGNED_SHORT_5_6_5             0x8363
#define GL_VERTEX_SHADER                    0x8B31
#define GL_ARRAY_BUFFER_BINDING             0x8894
#define GL_READ_FRAMEBUFFER                 0x8CA8
#define GL_DRAW_FRAMEBUFFER                 0x8CA9
#define GL_TEXTURE_MAX_ANISOTROPY_EXT       0x84FE
#define GL_LUMINANCE32UI_EXT                0x8D74
#define GL_LUMINANCE_INTEGER_EXT            0x8D9C
#define GL_DEPTH_STENCIL_EXT                0x84F9
#define GL_RGBA16F                          0x881A
#define GL_R32F                             0x822E
#define GL_RG                               0x8227
#define GL_R16F                             0x822D
#define GL_RG16F                            0x822F
#define GL_RGBA32UI_EXT                     0x8D70
#define GL_RGBA_INTEGER_EXT                 0x8D99
#define GL_R16UI                            0x8234
#define GL_RG_INTEGER                       0x8228
#define GL_DEPTH_COMPONENT32                0x81A7
#define GL_DEPTH_COMPONENT32F               0x8CAC
#define GL_DEPTH_COMPONENT16                0x81A5
#define GL_DEPTH_COMPONENT24                0x81A6
#define GL_DEPTH24_STENCIL8_EXT             0x88F0
#define GL_DEPTH_STENCIL_EXT                0x84F9
#define GL_LUMINANCE32F_ARB                 0x8818
#define GL_TEXTURE_RENDERBUFFER_NV          0x8E55
#define GL_RENDERBUFFER_EXT                 0x8D41
#define GL_RENDERBUFFER_COVERAGE_SAMPLES_NV 0x8CAB
#define GL_RENDERBUFFER_COLOR_SAMPLES_NV    0x8E10

#define WGL_ACCELERATION_ARB                0x2003
#define WGL_ACCUM_BITS_ARB                  0x201D
#define WGL_ALPHA_BITS_ARB                  0x201B
#define WGL_AUX_BUFFERS_ARB                 0x2024
#define WGL_BLUE_BITS_ARB                   0x2019
#define WGL_DEPTH_BITS_ARB                  0x2022
#define WGL_DOUBLE_BUFFER_ARB               0x2011
#define WGL_DRAW_TO_WINDOW_ARB              0x2001
#define WGL_FULL_ACCELERATION_ARB           0x2027
#define WGL_GREEN_BITS_ARB                  0x2017
#define WGL_PIXEL_TYPE_ARB                  0x2013
#define WGL_RED_BITS_ARB                    0x2015
#define WGL_SAMPLES_ARB                     0x2042
#define WGL_STENCIL_BITS_ARB                0x2023
#define WGL_STEREO_ARB                      0x2012
#define WGL_SUPPORT_OPENGL_ARB              0x2010
#define WGL_TYPE_RGBA_ARB                   0x202B
#define WGL_NUMBER_OVERLAYS_ARB             0x2008
#define WGL_NUMBER_UNDERLAYS_ARB            0x2009

#endif

//------------------------------------------------------------------------
// GL_NV_path_rendering
//------------------------------------------------------------------------

#define GL_CLOSE_PATH_NV                                    0x00
#define GL_MOVE_TO_NV                                       0x02
#define GL_RELATIVE_MOVE_TO_NV                              0x03
#define GL_LINE_TO_NV                                       0x04
#define GL_RELATIVE_LINE_TO_NV                              0x05
#define GL_HORIZONTAL_LINE_TO_NV                            0x06
#define GL_RELATIVE_HORIZONTAL_LINE_TO_NV                   0x07
#define GL_VERTICAL_LINE_TO_NV                              0x08
#define GL_RELATIVE_VERTICAL_LINE_TO_NV                     0x09
#define GL_QUADRATIC_CURVE_TO_NV                            0x0A
#define GL_RELATIVE_QUADRATIC_CURVE_TO_NV                   0x0B
#define GL_CUBIC_CURVE_TO_NV                                0x0C
#define GL_RELATIVE_CUBIC_CURVE_TO_NV                       0x0D
#define GL_SMOOTH_QUADRATIC_CURVE_TO_NV                     0x0E
#define GL_RELATIVE_SMOOTH_QUADRATIC_CURVE_TO_NV            0x0F
#define GL_SMOOTH_CUBIC_CURVE_TO_NV                         0x10
#define GL_RELATIVE_SMOOTH_CUBIC_CURVE_TO_NV                0x11
#define GL_SMALL_CCW_ARC_TO_NV                              0x12
#define GL_RELATIVE_SMALL_CCW_ARC_TO_NV                     0x13
#define GL_SMALL_CW_ARC_TO_NV                               0x14
#define GL_RELATIVE_SMALL_CW_ARC_TO_NV                      0x15
#define GL_LARGE_CCW_ARC_TO_NV                              0x16
#define GL_RELATIVE_LARGE_CCW_ARC_TO_NV                     0x17
#define GL_LARGE_CW_ARC_TO_NV                               0x18
#define GL_RELATIVE_LARGE_CW_ARC_TO_NV                      0x19
#define GL_CIRCULAR_CCW_ARC_TO_NV                           0xF8
#define GL_CIRCULAR_CW_ARC_TO_NV                            0xFA
#define GL_CIRCULAR_TANGENT_ARC_TO_NV                       0xFC
#define GL_ARC_TO_NV                                        0xFE
#define GL_RELATIVE_ARC_TO_NV                               0xFF
#define GL_PATH_FORMAT_SVG_NV                               0x9070
#define GL_PATH_FORMAT_PS_NV                                0x9071
#define GL_STANDARD_FONT_NAME_NV                            0x9072
#define GL_SYSTEM_FONT_NAME_NV                              0x9073
#define GL_FILE_NAME_NV                                     0x9074
#define GL_PATH_STROKE_WIDTH_NV                             0x9075
#define GL_PATH_END_CAPS_NV                                 0x9076
#define GL_PATH_INITIAL_END_CAP_NV                          0x9077
#define GL_PATH_TERMINAL_END_CAP_NV                         0x9078
#define GL_PATH_JOIN_STYLE_NV                               0x9079
#define GL_PATH_MITER_LIMIT_NV                              0x907A
#define GL_PATH_DASH_CAPS_NV                                0x907B
#define GL_PATH_INITIAL_DASH_CAP_NV                         0x907C
#define GL_PATH_TERMINAL_DASH_CAP_NV                        0x907D
#define GL_PATH_DASH_OFFSET_NV                              0x907E
#define GL_PATH_CLIENT_LENGTH_NV                            0x907F
#define GL_PATH_FILL_MODE_NV                                0x9080
#define GL_PATH_FILL_MASK_NV                                0x9081
#define GL_PATH_FILL_COVER_MODE_NV                          0x9082
#define GL_PATH_STROKE_COVER_MODE_NV                        0x9083
#define GL_PATH_STROKE_MASK_NV                              0x9084
#define GL_PATH_SAMPLE_QUALITY_NV                           0x9085
#define GL_COUNT_UP_NV                                      0x9088
#define GL_COUNT_DOWN_NV                                    0x9089
#define GL_PATH_OBJECT_BOUNDING_BOX_NV                      0x908A
#define GL_CONVEX_HULL_NV                                   0x908B
#define GL_BOUNDING_BOX_NV                                  0x908D
#define GL_TRANSLATE_X_NV                                   0x908E
#define GL_TRANSLATE_Y_NV                                   0x908F
#define GL_TRANSLATE_2D_NV                                  0x9090
#define GL_TRANSLATE_3D_NV                                  0x9091
#define GL_AFFINE_2D_NV                                     0x9092
#define GL_AFFINE_3D_NV                                     0x9094
#define GL_TRANSPOSE_AFFINE_2D_NV                           0x9096
#define GL_TRANSPOSE_AFFINE_3D_NV                           0x9098
#define GL_UTF8_NV                                          0x909A
#define GL_UTF16_NV                                         0x909B
#define GL_BOUNDING_BOX_OF_BOUNDING_BOXES_NV                0x909C
#define GL_PATH_COMMAND_COUNT_NV                            0x909D
#define GL_PATH_COORD_COUNT_NV                              0x909E
#define GL_PATH_DASH_ARRAY_COUNT_NV                         0x909F
#define GL_PATH_COMPUTED_LENGTH_NV                          0x90A0
#define GL_PATH_FILL_BOUNDING_BOX_NV                        0x90A1
#define GL_PATH_STROKE_BOUNDING_BOX_NV                      0x90A2
#define GL_SQUARE_NV                                        0x90A3
#define GL_ROUND_NV                                         0x90A4
#define GL_TRIANGULAR_NV                                    0x90A5
#define GL_BEVEL_NV                                         0x90A6
#define GL_MITER_REVERT_NV                                  0x90A7
#define GL_MITER_TRUNCATE_NV                                0x90A8
#define GL_SKIP_MISSING_GLYPH_NV                            0x90A9
#define GL_USE_MISSING_GLYPH_NV                             0x90AA
#define GL_PATH_DASH_OFFSET_RESET_NV                        0x90B4
#define GL_MOVE_TO_RESETS_NV                                0x90B5
#define GL_MOVE_TO_CONTINUES_NV                             0x90B6
#define GL_BOLD_BIT_NV                                      0x01
#define GL_ITALIC_BIT_NV                                    0x02
#define GL_PATH_ERROR_POSITION_NV                           0x90AB
#define GL_PATH_FOG_GEN_MODE_NV                             0x90AC
#define GL_GLYPH_WIDTH_BIT_NV                               0x01
#define GL_GLYPH_HEIGHT_BIT_NV                              0x02
#define GL_GLYPH_HORIZONTAL_BEARING_X_BIT_NV                0x04
#define GL_GLYPH_HORIZONTAL_BEARING_Y_BIT_NV                0x08
#define GL_GLYPH_HORIZONTAL_BEARING_ADVANCE_BIT_NV          0x10
#define GL_GLYPH_VERTICAL_BEARING_X_BIT_NV                  0x20
#define GL_GLYPH_VERTICAL_BEARING_Y_BIT_NV                  0x40
#define GL_GLYPH_VERTICAL_BEARING_ADVANCE_BIT_NV            0x80
#define GL_GLYPH_HAS_KERNING_NV                             0x100
#define GL_FONT_X_MIN_BOUNDS_NV                             0x00010000
#define GL_FONT_Y_MIN_BOUNDS_NV                             0x00020000
#define GL_FONT_X_MAX_BOUNDS_NV                             0x00040000
#define GL_FONT_Y_MAX_BOUNDS_NV                             0x00080000
#define GL_FONT_UNITS_PER_EM_NV                             0x00100000
#define GL_FONT_ASCENDER_NV                                 0x00200000
#define GL_FONT_DESCENDER_NV                                0x00400000
#define GL_FONT_HEIGHT_NV                                   0x00800000
#define GL_FONT_MAX_ADVANCE_WIDTH_NV                        0x01000000
#define GL_FONT_MAX_ADVANCE_HEIGHT_NV                       0x02000000
#define GL_FONT_UNDERLINE_POSITION_NV                       0x04000000
#define GL_FONT_UNDERLINE_THICKNESS_NV                      0x08000000
#define GL_FONT_HAS_KERNING_NV                              0x10000000
#define GL_ACCUM_ADJACENT_PAIRS_NV                          0x90AD
#define GL_ADJACENT_PAIRS_NV                                0x90AE
#define GL_FIRST_TO_REST_NV                                 0x90AF
#define GL_PATH_GEN_MODE_NV                                 0x90B0
#define GL_PATH_GEN_COEFF_NV                                0x90B1
#define GL_PATH_GEN_COLOR_FORMAT_NV                         0x90B2
#define GL_PATH_GEN_COMPONENTS_NV                           0x90B3
#define GL_PATH_STENCIL_FUNC_NV                             0x90B7
#define GL_PATH_STENCIL_REF_NV                              0x90B8
#define GL_PATH_STENCIL_VALUE_MASK_NV                       0x90B9
#define GL_PATH_STENCIL_DEPTH_OFFSET_FACTOR_NV              0x90BD
#define GL_PATH_STENCIL_DEPTH_OFFSET_UNITS_NV               0x90BE
#define GL_PATH_COVER_DEPTH_FUNC_NV                         0x90BF

//------------------------------------------------------------------------

#if (!FW_CUDA)
#   define FW_DLL_IMPORT_RETV(RET, CALL, NAME, PARAMS, PASS)        bool isAvailable_ ## NAME(void);
#   define FW_DLL_IMPORT_VOID(RET, CALL, NAME, PARAMS, PASS)        bool isAvailable_ ## NAME(void);
#   define FW_DLL_DECLARE_RETV(RET, CALL, NAME, PARAMS, PASS)       bool isAvailable_ ## NAME(void); RET CALL NAME PARAMS;
#   define FW_DLL_DECLARE_VOID(RET, CALL, NAME, PARAMS, PASS)       bool isAvailable_ ## NAME(void); RET CALL NAME PARAMS;
#   if (FW_USE_CUDA)
#       define FW_DLL_IMPORT_CUDA(RET, CALL, NAME, PARAMS, PASS)    bool isAvailable_ ## NAME(void);
#       define FW_DLL_IMPORT_CUV2(RET, CALL, NAME, PARAMS, PASS)    bool isAvailable_ ## NAME(void);
#   else
#       define FW_DLL_IMPORT_CUDA(RET, CALL, NAME, PARAMS, PASS)    bool isAvailable_ ## NAME(void); RET CALL NAME PARAMS;
#       define FW_DLL_IMPORT_CUV2(RET, CALL, NAME, PARAMS, PASS)    bool isAvailable_ ## NAME(void); RET CALL NAME PARAMS;
#   endif
#   include "base/DLLImports.inl"
#   undef FW_DLL_IMPORT_RETV
#   undef FW_DLL_IMPORT_VOID
#   undef FW_DLL_DECLARE_RETV
#   undef FW_DLL_DECLARE_VOID
#   undef FW_DLL_IMPORT_CUDA
#   undef FW_DLL_IMPORT_CUV2
#endif

//------------------------------------------------------------------------
