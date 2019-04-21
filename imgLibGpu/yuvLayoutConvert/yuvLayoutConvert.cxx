#if HOSTCODE
    #include "yuvLayoutConvert.h"
#endif

#include "gpuSupport/gpuTool.h"
#include "gpuDevice/loadstore/loadNorm.h"
#include "gpuDevice/loadstore/storeNorm.h"
#include "vectorTypes/vectorOperations.h"
#include "yuvLayoutConvert/yuvLayoutConvertCommon.h"

#if HOSTCODE
    #include "yuvLayoutConvert/yuv420Tools.h"
#endif

namespace yuvLayoutConvert {

//================================================================
//
// importLumaData
//
//================================================================

GPUTOOL_2D_BEG
(
    importLumaData,
    PREP_EMPTY,
    ((Luma, dst)),
    ((GpuArray<const uint8>, data))
)
#if DEVCODE
{
    ARRAY_EXPOSE(data);

    Space dataOfs = X + Y * vGlobSize.X;

    float32 value = 0;

    if (SpaceU(dataOfs) < SpaceU(dataSize))
        value = loadNorm(&dataPtr[dataOfs]);

    COMPILE_ASSERT(TYPE_IS_SIGNED(Luma));
    storeNorm(dst, 2 * value - 1); // to range [-1, +1]
}
#endif
GPUTOOL_2D_END

//================================================================
//
// importChromaData
//
//================================================================

GPUTOOL_2D_BEG
(
    importChromaData,
    PREP_EMPTY,
    ((Chroma, dst)),
    ((GpuArray<const uint8>, data))
)
#if DEVCODE
{
    ARRAY_EXPOSE(data);

    Point<Space> chromaSize = vGlobSize;
    Space chromaArea = chromaSize.X * chromaSize.Y;
    Space lumaArea = chromaArea << 2;

    ////

    Space chromaOfs = X + Y * vGlobSize.X;

    Space chromaOfsU = lumaArea + chromaOfs;
    Space chromaOfsV = chromaOfsU + chromaArea;

    ////

    float32 chromaU = 0;

    if (SpaceU(chromaOfsU) < SpaceU(dataSize))
        chromaU = loadNorm(&dataPtr[chromaOfsU]);

    ////

    float32 chromaV = 0;

    if (SpaceU(chromaOfsV) < SpaceU(dataSize))
        chromaV = loadNorm(&dataPtr[chromaOfsV]);

    ////

    COMPILE_ASSERT(TYPE_IS_SIGNED(Chroma));
    storeNorm(dst, 2 * make_float32_x2(chromaU, chromaV) - 1);
}
#endif
GPUTOOL_2D_END

//================================================================
//
// convertRawToYuv420
//
//================================================================

#if HOSTCODE

template <>
stdbool convertRawToYuv420(const GpuArray<const uint8>& src, const GpuImageYuv<Luma>& dst, stdPars(GpuProcessKit))
{
    stdBegin;

    ////

    Point<Space> imageSize = dst.luma.size();
    REQUIRE(yuv420SizeValid(imageSize));
    REQUIRE(src.size() == yuv420TotalArea(imageSize));

    ////

    require(importLumaData(dst.luma, src, stdPass));
    require(importChromaData(dst.chroma, src, stdPass));

    ////

    stdEnd;
}

#endif

//================================================================
//
// convertRawToYuv420
//
//================================================================

#if HOSTCODE

template <>
stdbool convertRawToYuv420(const GpuArray<const uint16>& src, const GpuImageYuv<Luma>& dst, stdPars(GpuProcessKit))
{
    stdBegin;
    REQUIRE(false); // not implemented
    stdEnd;
}

#endif

//================================================================
//
// exportLumaData
//
//================================================================

GPUTOOL_2D_BEG
(
    exportLumaData,
    PREP_EMPTY,
    ((const Luma, src)),
    ((GpuArray<uint8>, data))
)
#if DEVCODE
{
    ARRAY_EXPOSE(data);

    Space lumaOfs = X + Y * vGlobSize.X;

    float32 value = 0.5f * loadNorm(src) + 0.5f; // to [0, 1]

    if (SpaceU(lumaOfs) < SpaceU(dataSize))
        storeNorm(&dataPtr[lumaOfs], value);
}
#endif
GPUTOOL_2D_END

//================================================================
//
// exportChromaData
//
//================================================================

GPUTOOL_2D_BEG
(
    exportChromaData,
    PREP_EMPTY,
    ((const Chroma, src)),
    ((GpuArray<uint8>, data))
)
#if DEVCODE
{
    ARRAY_EXPOSE(data);

    Point<Space> chromaSize = vGlobSize;
    Space chromaArea = chromaSize.X * chromaSize.Y;
    Space lumaArea = chromaArea << 2;

    ////

    Space chromaOfs = X + Y * vGlobSize.X;

    Space chromaOfsU = lumaArea + chromaOfs;
    Space chromaOfsV = chromaOfsU + chromaArea;

    ////

    float32_x2 value = 0.5f * loadNorm(src) + 0.5f; // to [0, 1]

    ////

    if (SpaceU(chromaOfsU) < SpaceU(dataSize))
        storeNorm(&dataPtr[chromaOfsU], value.x);

    if (SpaceU(chromaOfsV) < SpaceU(dataSize))
        storeNorm(&dataPtr[chromaOfsV], value.y);
}
#endif
GPUTOOL_2D_END

//================================================================
//
// convertYuv420ToRaw
//
//================================================================

#if HOSTCODE

template <>
stdbool convertYuv420ToRaw(const GpuImageYuv<const Luma>& src, const GpuArray<uint8>& dst, stdPars(GpuProcessKit))
{
    stdBegin;

    ////

    Point<Space> imageSize = src.luma.size();
    REQUIRE(yuv420SizeValid(imageSize));
    REQUIRE(dst.size() == yuv420TotalArea(imageSize));

    ////

    require(exportLumaData(src.luma, dst, stdPass));
    require(exportChromaData(src.chroma, dst, stdPass));

    ////

    stdEnd;
}

#endif

//================================================================
//
// convertYuv420ToRaw
//
//================================================================

#if HOSTCODE

template <>
stdbool convertYuv420ToRaw(const GpuImageYuv<const Luma>& src, const GpuArray<uint16>& dst, stdPars(GpuProcessKit))
{
    stdBegin;

    REQUIRE(false); // not implemented

    stdEnd;
}

#endif

//----------------------------------------------------------------

}

