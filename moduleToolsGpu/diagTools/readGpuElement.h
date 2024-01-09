#pragma once

#include "gpuProcessHeader.h"
#include "dataAlloc/matrixMemory.h"
#include "gpuAppliedApi/gpuAppliedApi.h"

//================================================================
//
// readGpuElement
//
//================================================================

template <typename Type, typename Pitch>
stdbool readGpuElement
(
    const GpuMatrix<const Type, Pitch>& image,
    const Point<Space>& pos,
    Type& result,
    stdPars(GpuProcessKit)
)
{
    GpuMatrix<const Type, Pitch> gpuElement;
    require(image.subs(pos, point(1), gpuElement));

    MatrixMemory<Type> cpuElement;
    require(cpuElement.reallocForGpuExch(point(1), stdPass));

    GpuCopyThunk gpuCopy;
    require(gpuCopy(gpuElement, cpuElement, stdPass));
    gpuCopy.waitClear();

    result = Type{};

    if (kit.dataProcessing)
    {
        MATRIX_EXPOSE(cpuElement);
        result = *cpuElementMemPtr;
    }

    returnTrue;
}
