#include "gpuPyramidCache.h"

#include "errorLog/errorLog.h"
#include "gpuAppliedApi/gpuAppliedApi.h"
#include "dataAlloc/arrayMemory.inl"

//================================================================
//
// GpuPyramidCache::realloc
//
//================================================================

void GpuPyramidCache::realloc(stdPars(GpuProcessKit))
{
    allocated = false;

    ////

    cpuHolder.realloc(1, kit.gpuProperties.samplerAndFastTransferBaseAlignment, stdPass);
    gpuHolder.realloc(1, kit.gpuProperties.samplerAndFastTransferBaseAlignment, stdPass);

    ////

    if (kit.dataProcessing)
    {
        ARRAY_EXPOSE(cpuHolder);
        initEmpty(helpModify(*cpuHolderPtr));
    }

    ////

    allocated = true;
}

//================================================================
//
// GpuPyramidCache::slowUpdate
//
//================================================================

void GpuPyramidCache::slowUpdate(const GpuPyramidLayout& layout, stdPars(GpuProcessKit))
{
    REQUIRE(allocated);

    ////

    ARRAY_EXPOSE(cpuHolder);
    *cpuHolderPtr = layout;

    ////

    GpuCopyThunk gpuCopy;
    gpuCopy(cpuHolder, gpuHolder, stdPass);
}
