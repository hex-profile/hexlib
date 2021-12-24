#include "copyImageFromCpu.h"

#include "dataAlloc/arrayMemoryStatic.h"
#include "dataAlloc/arrayObjectMemoryStatic.h"
#include "dataAlloc/gpuMatrixMemory.h"
#include "flipMatrix.h"
#include "gpuAppliedApi/gpuAppliedApi.h"

namespace packageImpl {

//================================================================
//
// copyImageFromCpu
//
//================================================================

template <typename Pixel>
stdbool copyImageFromCpu
(
    const Matrix<const Pixel> srcImage,
    GpuArrayMemory<Pixel>& memory,
    GpuMatrix<const Pixel>& dst,
    GpuCopyThunk& gpuCopier,
    stdPars(GpuProcessKit)
)
{
    auto src = srcImage;

    ////

    bool inverted = false;

    if (src.memPitch() < 0)
        {src = flipMatrix(src); inverted = true;}

    ////

    MATRIX_EXPOSE_UNSAFE(src);

    Array<const Pixel> srcArray;
    REQUIRE(srcMemPitch >= srcSizeX);
    srcArray.assign(srcMemPtr, srcMemPitch * srcSizeY);

    auto& dstArray = memory;
    require(dstArray.realloc(srcArray.size(), stdPass));

    ////

    require(gpuCopier(srcArray, dstArray, stdPass));

    ////

    REQUIRE(dst.assign(dstArray.ptr(), srcMemPitch, srcSizeX, srcSizeY));

    if (inverted)
        dst = flipMatrix(dst);

    ////

    returnTrue;
}

//----------------------------------------------------------------

INSTANTIATE_FUNC(copyImageFromCpu<uint8>);
INSTANTIATE_FUNC(copyImageFromCpu<uint8_x4>);

//----------------------------------------------------------------

}
