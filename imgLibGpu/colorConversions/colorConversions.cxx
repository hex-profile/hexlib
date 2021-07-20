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

GPUTOOL_2D_BEG
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
    convertBgrToYPbPr(loadNorm(src), Y, Pb, Pr);
    storeNorm(dst, 0.5f * Y + 0.5f); // from [-1, +1] to [0, 1].
}
#endif    
GPUTOOL_2D_END

//================================================================
//
// convertBgr32ToBgr24
//
//================================================================

GPUTOOL_2D_BEG
(
    convertBgr32ToBgr24,
    PREP_EMPTY,
    ((const uint8_x4, src)),
    ((GpuMatrix<uint8>, dst))
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

//================================================================
//
// convertBgr24ToBgr32Kernel
//
//================================================================

GPUTOOL_2D_BEG
(
    convertBgr24ToBgr32Kernel,
    PREP_EMPTY,
    ((uint8_x4, dst)),
    ((GpuMatrix<const uint8>, src))
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
