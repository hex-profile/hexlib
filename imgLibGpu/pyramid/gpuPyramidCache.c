#include "gpuPyramidCache.h"

#include "errorLog/errorLog.h"
#include "gpuAppliedApi/gpuAppliedApi.h"
#include "dataAlloc/arrayMemory.inl"

//================================================================
//
// GpuPyramidCache::realloc
//
//================================================================

stdbool GpuPyramidCache::realloc(stdPars(GpuProcessKit))
{
    allocated = false;

    ////

    require(cpuHolder.realloc(1, kit.gpuProperties.samplerBaseAlignment, stdPass));
    require(gpuHolder.realloc(1, kit.gpuProperties.samplerBaseAlignment, stdPass));

    ////

    if (kit.dataProcessing)
    {
        ARRAY_EXPOSE(cpuHolder);
        initEmpty(helpModify(*cpuHolderPtr));
    }

    ////

    allocated = true;

    returnTrue;
}

//================================================================
//
// GpuPyramidCache::slowUpdate
//
//================================================================

stdbool GpuPyramidCache::slowUpdate(const GpuPyramidLayout& layout, stdPars(GpuProcessKit))
{
    REQUIRE(allocated);

    ////

    ARRAY_EXPOSE(cpuHolder);
    *cpuHolderPtr = layout;

    ////

    GpuCopyThunk gpuCopy;
    require(gpuCopy(cpuHolder, gpuHolder, stdPass));

    ////

    returnTrue;
}
