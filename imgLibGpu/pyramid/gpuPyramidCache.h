#pragma once

#include "gpuPyramidLayout.h"
#include "gpuProcessHeader.h"
#include "pyramid/gpuPyramid.h"
#include "dataAlloc/gpuArrayMemory.h"
#include "errorLog/errorLog.h"

//================================================================
//
// GpuPyramidCache
//
//================================================================

class GpuPyramidCache
{

public:

    void realloc(stdPars(GpuProcessKit));

    sysinline void dealloc()
    {
        cpuHolder.dealloc();
        gpuHolder.dealloc();
    }

public:

    template <typename Type>
    void getDevicePyramid(const GpuPyramid<Type>& pyramid, GpuPyramidParam<Type>& result, stdPars(GpuProcessKit))
    {
        REQUIRE(allocated);

        ////

        GpuPyramidLayout tmpLayout;
        REQUIRE(pyramid.getGpuLayout(result.basePointer, tmpLayout));

        ////

        ARRAY_EXPOSE(gpuHolder);
        ARRAY_EXPOSE(cpuHolder);

        result.gpuLayout = unsafePtr(gpuHolderPtr, 1);
        result.cpuLayout = unsafePtr(cpuHolderPtr, 1);

        result.levels = tmpLayout.levels;
        result.layers = tmpLayout.layers;

        ////

        if_not (kit.dataProcessing)
            return;

        ////

        if_not (isEqualLayout(tmpLayout, *cpuHolderPtr))
            slowUpdate(tmpLayout, stdPass);
    }

private:

    void slowUpdate(const GpuPyramidLayout& layout, stdPars(GpuProcessKit));

private:

    bool allocated = false;
    ArrayMemory<GpuPyramidLayout> cpuHolder;
    GpuArrayMemory<GpuPyramidLayout> gpuHolder;

};
