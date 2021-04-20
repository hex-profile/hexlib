#include "conversions.h"

#include "gpuDevice/loadstore/loadNorm.h"
#include "gpuDevice/loadstore/storeNorm.h"
#include "gpuSupport/gpuTool.h"

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
