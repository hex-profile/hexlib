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
void copyImageFromCpu
(
    const MatrixAP<const Pixel> srcImage,
    GpuArrayMemory<Pixel>& memory,
    GpuMatrixAP<const Pixel>& dst,
    GpuCopyThunk& gpuCopier,
    stdPars(GpuProcessKit)
)
{
    auto src = relaxToAnyPitch(srcImage);

    ////

    bool inverted = false;

    if (src.memPitch() < 0)
        {src = flipMatrix(src); inverted = true;}

    ////

    MATRIX_EXPOSE_UNSAFE(src);

    REQUIRE(srcMemPitch >= srcSizeX);

    Array<const Pixel> srcArray;
    REQUIRE(srcArray.assignValidated(srcMemPtr, srcMemPitch * srcSizeY));

    auto& dstArray = memory;
    dstArray.realloc(srcArray.size(), stdPass);

    ////

    gpuCopier(srcArray, dstArray, stdPass);

    ////

    REQUIRE(dst.assignValidated(dstArray.ptrUnsafeForInternalUseOnly(), srcMemPitch, srcSizeX, srcSizeY));

    if (inverted)
        dst = flipMatrix(dst);
}

////

INSTANTIATE_FUNC(copyImageFromCpu<uint8>)
INSTANTIATE_FUNC(copyImageFromCpu<uint8_x4>)

//----------------------------------------------------------------

}
