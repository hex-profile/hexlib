#pragma once

#include "dataAlloc/gpuArrayMemory.h"
#include "gpuProcessHeader.h"
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
);

//----------------------------------------------------------------

}
