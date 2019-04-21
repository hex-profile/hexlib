#pragma once

#include "numbers/float/floatBase.h"
#include "numbers/int/intBase.h"
#include "numbers/interface/numberInterface.h"
#include "stdFunc/stdFunc.h"
#include "data/gpuArray.h"
#include "data/gpuMatrix.h"
#include "vectorTypes/vectorType.h"
#include "imageRead/borderMode.h"
#include "gpuAppliedApi/gpuApiBasics.h"

//================================================================
//
// GpuChannelType
//
//================================================================

enum GpuChannelType
{
    GpuChannelInt8,
    GpuChannelUint8,

    GpuChannelInt16,
    GpuChannelUint16,

    GpuChannelInt32,
    GpuChannelUint32,

    GpuChannelFloat16,
    GpuChannelFloat32,

    GpuChannelTypeCount
};

//================================================================
//
// GPU_CHANTYPE_FOREACH
//
//================================================================

#define GPU_CHANTYPE_FOREACH(action, extra) \
    action(GpuChannelInt8, int8, extra) \
    action(GpuChannelUint8, uint8, extra) \
    action(GpuChannelInt16, int16, extra) \
    action(GpuChannelUint16, uint16, extra) \
    action(GpuChannelInt32, int32, extra) \
    action(GpuChannelUint32, uint32, extra) \
    action(GpuChannelFloat16, float16, extra) \
    action(GpuChannelFloat32, float32, extra)

//================================================================
//
// GpuGetScalarChannelType
//
//================================================================

template <typename Type>
struct GpuGetScalarChannelType;

#define TMP_MACRO(chanType, Type, o) \
    \
    template <> \
    struct GpuGetScalarChannelType<Type> \
    { \
        static const GpuChannelType val = chanType; \
    };

GPU_CHANTYPE_FOREACH(TMP_MACRO, o)

#undef TMP_MACRO

//================================================================
//
// GpuGetChannelType
//
//================================================================

template <typename Type>
struct GpuGetChannelType
{
    static const GpuChannelType val = GpuGetScalarChannelType<VECTOR_BASE(Type)>::val;
};

//================================================================
//
// GpuSamplerLink
//
//================================================================

struct GpuSamplerLink;

//================================================================
//
// GpuSamplerSetup
//
//================================================================

struct GpuSamplerSetup
{

    //
    // setSamplerArray
    //

    virtual bool setSamplerArray
    (
        const GpuSamplerLink& sampler,
        GpuAddrU arrayAddr,
        Space arrayByteSize,
        GpuChannelType chanType,
        int rank,
        BorderMode borderMode,
        bool linearInterpolation,
        bool readNormalizedFloat,
        bool normalizedCoords,
        const GpuContext& context,
        stdNullPars
    )
    =0;

    template <typename Type, typename Kit>
    inline stdbool setSamplerArray
    (
        const GpuSamplerLink& sampler,
        const GpuArray<const Type>& array,
        BorderMode borderMode,
        bool linearInterpolation,
        bool readNormalizedFloat,
        bool normalizedCoords,
        stdPars(Kit)
    )
    {
        return setSamplerArray
        (
            sampler,
            GpuAddrU(array.ptrUnsafeForInternalUseOnly()),
            array.size() * sizeof(Type),
            GpuGetChannelType<Type>::val,
            VectorTypeRank<Type>::val,
            borderMode,
            linearInterpolation,
            readNormalizedFloat,
            normalizedCoords,
            kit.gpuCurrentContext,
            stdPassThru
        );
    }

    //
    // setSamplerImage
    //

    virtual bool setSamplerImage
    (
        const GpuSamplerLink& sampler,
        GpuAddrU imageBaseAddr,
        Space imageBytePitch,
        const Point<Space>& imageSize,
        GpuChannelType chanType,
        int rank,
        BorderMode borderMode,
        bool linearInterpolation,
        bool readNormalizedFloat,
        bool normalizedCoords,
        const GpuContext& context,
        stdNullPars
    )
    =0;

    template <typename Type, typename Kit>
    inline stdbool setSamplerImage
    (
        const GpuSamplerLink& sampler,
        const GpuMatrix<const Type>& image,
        BorderMode borderMode,
        bool linearInterpolation,
        bool readNormalizedFloat,
        bool normalizedCoords,
        stdPars(Kit)
    )
    {
        return setSamplerImage
        (
            sampler,
            GpuAddrU(image.memPtrUnsafeInternalUseOnly()),
            image.memPitch() * sizeof(Type),
            image.size(),
            GpuGetChannelType<Type>::val,
            VectorTypeRank<Type>::val,
            borderMode,
            linearInterpolation,
            readNormalizedFloat,
            normalizedCoords,
            kit.gpuCurrentContext,
            stdPassThru
        );
    }

};
