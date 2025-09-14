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
void readGpuElement
(
    const GpuMatrix<const Type, Pitch>& image,
    const Point<Space>& pos,
    Type& result,
    stdPars(GpuProcessKit)
)
{
    GpuMatrix<const Type, Pitch> gpuElement;
    REQUIRE(image.subs(pos, point(1), gpuElement));

    MatrixMemory<Type> cpuElement;
    cpuElement.reallocForGpuExch(point(1), stdPass);

    GpuCopyThunk gpuCopy;
    gpuCopy(gpuElement, cpuElement, stdPass);
    gpuCopy.waitClear();

    result = Type{};

    if (kit.dataProcessing)
    {
        MATRIX_EXPOSE(cpuElement);
        result = *cpuElementMemPtr;
    }
}
