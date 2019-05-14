#pragma once

#include "gpuPyramidLayout.h"
#include "gpuProcessHeader.h"
#include "pyramid/gpuPyramid.h"
#include "dataAlloc/gpuArrayMemory.h"

//================================================================
//
// GpuPyramidCache
//
//================================================================

class GpuPyramidCache
{

public:

    stdbool realloc(stdPars(GpuProcessKit));

    sysinline void dealloc()
    {
        cpuHolder.dealloc();
        gpuHolder.dealloc();
    }

public:

    template <typename Type>
    stdbool getDevicePyramid(const GpuPyramid<Type>& pyramid, GpuPyramidParam<Type>& result, stdPars(GpuProcessKit))
    {
        stdBegin;

        REQUIRE(allocated);

        ////

        GpuPyramidLayout tmpLayout;
        REQUIRE(pyramid.getGpuLayout(result.basePointer, tmpLayout));

        ////

        ARRAY_EXPOSE(gpuHolder);
        ARRAY_EXPOSE(cpuHolder);

        result.gpuLayout = unsafePtr(gpuHolderPtr, 1);
        result.cpuLayout = unsafePtr(cpuHolderPtr, 1);

        result.levelCount = tmpLayout.levelCount;
        result.layerCount = tmpLayout.layerCount;

        ////

        if_not (kit.dataProcessing)
            return true;

        ////

        if_not (isEqualLayout(tmpLayout, *cpuHolderPtr))
            require(slowUpdate(tmpLayout, stdPass));

        ////

        stdEnd;
    }

private:

    stdbool slowUpdate(const GpuPyramidLayout& layout, stdPars(GpuProcessKit));

private:

    bool allocated = false;
    ArrayMemory<GpuPyramidLayout> cpuHolder;
    GpuArrayMemory<GpuPyramidLayout> gpuHolder;

};
