#pragma once

#include "imageRead/borderMode.h"
#include "devSamplerInterface.h"
#include "vectorTypes/vectorType.h"

#ifdef __CUDA_ARCH__

//================================================================
//
// CudaSamplerType
//
//================================================================

template <DevSamplerType samplerType>
struct CudaSamplerType
    {};

template <>
struct CudaSamplerType<DevSampler1D>
    {static const int val = cudaTextureType1D;};

template <>
struct CudaSamplerType<DevSampler2D>
    {static const int val = cudaTextureType2D;};

//================================================================
//
// CudaReadMode
//
//================================================================

template <DevSamplerReadMode readMode>
struct CudaReadMode
{
    static const cudaTextureReadMode val =
        readMode == DevSamplerFloat ?
        cudaReadModeNormalizedFloat :
        cudaReadModeElementType;
};

//================================================================
//
// CudaSamplerDeclType
//
// The type for CUDA sampler declaration:
// in CUDA, it has influence on template tex2D-like functions.
//
//================================================================

template <DevSamplerReadMode readMode, int rank>
struct CudaSamplerDeclType;

////

#define TMP_MACRO(readMode, rank, Type) \
    template <> struct CudaSamplerDeclType<readMode, rank> {using T = Type;};

////

TMP_MACRO(DevSamplerFloat, 1, int8)
TMP_MACRO(DevSamplerFloat, 2, int8_x2)
TMP_MACRO(DevSamplerFloat, 4, int8_x4)

TMP_MACRO(DevSamplerInt, 1, int32)
TMP_MACRO(DevSamplerInt, 2, int32_x2)
TMP_MACRO(DevSamplerInt, 4, int32_x4)

TMP_MACRO(DevSamplerUint, 1, uint32)
TMP_MACRO(DevSamplerUint, 2, uint32_x2)
TMP_MACRO(DevSamplerUint, 4, uint32_x4)

////

#undef TMP_MACRO

//================================================================
//
// devDefineSampler
//
//================================================================

#define devDefineSampler(sampler, samplerType, readMode, rank) \
    extern "C" {texture<CudaSamplerDeclType<readMode, rank>::T, CudaSamplerType<samplerType>::val, CudaReadMode<readMode>::val> sampler;}

//----------------------------------------------------------------

#define devSamplerParamType(samplerType, readMode, rank) \
    texture<CudaSamplerDeclType<readMode, rank>::T, CudaSamplerType<samplerType>::val, CudaReadMode<readMode>::val>

//================================================================
//
// DevSamplerResult<texture>
//
//================================================================

template <typename DeclType, int dim, cudaTextureReadMode texReadMode>
struct DevSamplerResult<texture<DeclType, dim, texReadMode>>
{
    static const int rank = VectorTypeRank<DeclType>::val;
    using BaseType = VECTOR_BASE(DeclType);

    static const DevSamplerReadMode elementReadMode =
        !TYPE_IS_BUILTIN_INT(BaseType) ? DevSamplerFloat :
        TYPE_IS_SIGNED(BaseType) ? DevSamplerInt : DevSamplerUint;

    static const DevSamplerReadMode readMode =
        texReadMode == cudaReadModeNormalizedFloat ? DevSamplerFloat : elementReadMode;

    using T = typename DevSamplerReturnType<readMode, rank>::T;
};

//================================================================
//
// devTex2D
//
//================================================================

template <typename SamplerType>
sysinline auto devTex2D(SamplerType sampler, float32 X, float32 Y)
{
    #pragma nv_diag_suppress 1215

    return tex2D(sampler, X, Y);

    #pragma nv_diag_default 1215
}

//================================================================
//
// devTex1Dfetch
//
//================================================================

template <typename SamplerType>
inline auto devTex1Dfetch(SamplerType sampler, int offset)
{
    return tex1Dfetch(sampler, offset)
}

//----------------------------------------------------------------

#endif // __CUDA_ARCH__
