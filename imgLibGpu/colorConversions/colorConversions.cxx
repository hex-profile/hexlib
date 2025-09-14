#include "colorConversions.h"

#include "convertYuv420/convertYuvRgbFunc.h"
#include "gpuDevice/loadstore/loadNorm.h"
#include "gpuDevice/loadstore/storeNorm.h"
#include "gpuSupport/gpuTool.h"

//================================================================
//
// convertBgr32ToMono
//
//================================================================

GPUTOOL_2D_BEG_AP
(
    convertBgr32ToMono,
    PREP_EMPTY,
    ((const uint8_x4, src))
    ((uint8, dst)),
    PREP_EMPTY
)
#if DEVCODE
{
    float32 Y, Pb, Pr;
    convertBgrToYPbPr<false>(loadNorm(src), Y, Pb, Pr);
    storeNorm(dst, Y);
}
#endif
GPUTOOL_2D_END

//================================================================
//
// convertMonoToBgr32
//
//================================================================

GPUTOOL_2D_BEG_AP
(
    convertMonoToBgr32,
    PREP_EMPTY,
    ((const uint8, src))
    ((uint8_x4, dst)),
    PREP_EMPTY
)
#if DEVCODE
{
    auto value = *src;
    *dst = make_uint8_x4(value, value, value, 0);
}
#endif
GPUTOOL_2D_END

//================================================================
//
// convertBgr32ToMonoBgr32
//
//================================================================

GPUTOOL_2D_BEG_AP
(
    convertBgr32ToMonoBgr32,
    PREP_EMPTY,
    ((const uint8_x4, src))
    ((uint8_x4, dst)),
    PREP_EMPTY
)
#if DEVCODE
{
    float32 Y, Pb, Pr;
    convertBgrToYPbPr<false>(loadNorm(src), Y, Pb, Pr);
    storeNorm(dst, make_float32_x4(Y, Y, Y, 0.f));
}
#endif
GPUTOOL_2D_END

//================================================================
//
// convertBgr32ToBgr24
//
//================================================================

GPUTOOL_2D_BEG_AP
(
    convertBgr32ToBgr24Func,
    PREP_EMPTY,
    ((const uint8_x4, src)),
    ((GpuMatrixAP<uint8>, dst))
)
#if DEVCODE
{
    MATRIX_EXPOSE(dst);

    ////

    auto monoX = 3 * X;
    auto monoY = Y;

    ////

    auto dstValid =
        monoX + 3 <= dstSizeX &&
        monoY < dstSizeY;

    if_not (dstValid)
        return;

    ////

    auto value = helpRead(*src);

    auto dstPtr = MATRIX_POINTER(dst, monoX, monoY);

    dstPtr[0] = value.x;
    dstPtr[1] = value.y;
    dstPtr[2] = value.z;
}
#endif
GPUTOOL_2D_END

//----------------------------------------------------------------

#if HOSTCODE

void convertBgr32ToBgr24(const GpuMatrixAP<const uint8_x4>& src, const GpuMatrixAP<uint8>& dst, stdPars(GpuProcessKit))
{
    REQUIRE(src.sizeX() * 3 == dst.sizeX());
    convertBgr32ToBgr24Func(src, dst, stdPassThru);
}

#endif

//================================================================
//
// convertBgr24ToBgr32
//
//================================================================

GPUTOOL_2D_BEG_AP
(
    convertBgr24ToBgr32Func,
    PREP_EMPTY,
    ((uint8_x4, dst)),
    ((GpuMatrixAP<const uint8>, src))
)
#if DEVCODE
{
    MATRIX_EXPOSE(src);

    auto monoX = 3 * X;
    auto monoY = Y;

    auto result = zeroOf<uint8_x4>();

    if (monoX + 3 <= srcSizeX && monoY < srcSizeY)
    {
        auto ptr = MATRIX_POINTER(src, monoX, monoY);
        result = make_uint8_x4(ptr[0], ptr[1], ptr[2], 0);
    }

    *dst = result;
}
#endif
GPUTOOL_2D_END

//----------------------------------------------------------------

#if HOSTCODE

void convertBgr24ToBgr32(const GpuMatrixAP<const uint8>& src, const GpuMatrixAP<uint8_x4>& dst, stdPars(GpuProcessKit))
{
    REQUIRE(src.sizeX() == dst.sizeX() * 3);
    convertBgr24ToBgr32Func(dst, src, stdPassThru);
}

#endif

//================================================================
//
// convertBgr24ToMono
//
//================================================================

GPUTOOL_2D_BEG_AP
(
    convertBgr24ToMonoFunc,
    PREP_EMPTY,
    ((uint8, dst)),
    ((GpuMatrixAP<const uint8>, src))
)
#if DEVCODE
{
    MATRIX_EXPOSE(src);

    auto monoX = 3 * X;
    auto monoY = Y;

    uint8 result = 0;

    if (monoX + 3 <= srcSizeX && monoY < srcSizeY)
    {
        auto ptr = MATRIX_POINTER(src, monoX, monoY);

        auto src = make_float32_x4(ptr[0], ptr[1], ptr[2], 0);
        float32 Y, Pb, Pr;
        convertBgrToYPbPr<false>(src, Y, Pb, Pr);
        result = convertRoundSaturate<uint8>(Y);
    }

    *dst = result;
}
#endif
GPUTOOL_2D_END

//----------------------------------------------------------------

#if HOSTCODE

void convertBgr24ToMono(const GpuMatrixAP<const uint8>& src, const GpuMatrixAP<uint8>& dst, stdPars(GpuProcessKit))
{
    REQUIRE(src.sizeX() == dst.sizeX() * 3);
    convertBgr24ToMonoFunc(dst, src, stdPassThru);
}

#endif
