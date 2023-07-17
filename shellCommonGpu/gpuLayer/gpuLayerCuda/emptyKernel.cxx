#include "emptyKernel.h"

#include "numbers/mathIntrinsics.h"
#include "gpuDevice/loadViaSamplerCache.h"

//================================================================
//
// emptyKernel
//
//================================================================

#if DEVCODE

devDefineKernel(emptyKernel, EmptyKernelParams, params)
{
}

#endif

//================================================================
//
// getEmptyKernelLink
//
//================================================================

#if HOSTCODE

const GpuKernelLink& getEmptyKernelLink()
{
    return emptyKernel;
}

#endif

//================================================================
//
// readMemoryKernel
//
//================================================================

#if DEVCODE

devDefineKernel(readMemoryKernel, ReadMemoryKernelParams, params)
{
    Space i = devThreadX + devGroupX * devThreadCountX;

    if_not (SpaceU(i) < params.atomCount)
        return;

    const ReadMemoryAtom* readPtr = params.srcDataPtr + i;

    ReadMemoryAtom value = loadViaSamplerCache(readPtr);

    if (params.writeEnabled)
        params.dstDataPtr[i] = value;
}

#endif

//================================================================
//
// getReadMemoryKernelLink
//
//================================================================

#if HOSTCODE

const GpuKernelLink& getReadMemoryKernelLink()
{
    return readMemoryKernel;
}

#endif
