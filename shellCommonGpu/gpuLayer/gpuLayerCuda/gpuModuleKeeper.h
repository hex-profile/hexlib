#pragma once

#include "errorLog/errorLog.h"
#include "gpuLayer/gpuLayer.h"
#include "dataAlloc/arrayObjectMemory.h"
#include "dataAlloc/memoryAllocatorKit.h"
#include "userOutput/errorLogEx.h"
#include "allocation/mallocKit.h"

//================================================================
//
// ModuleInfo
//
//================================================================

using ModuleInfo = GpuModuleOwner;

//================================================================
//
// KernelInfo
//
//================================================================

struct KernelInfo
{
    const char* name;
    GpuKernelOwner owner;
};

//================================================================
//
// SamplerInfo
//
//================================================================

struct SamplerInfo
{
    const char* name;
    GpuSamplerOwner owner;
};

//================================================================
//
// GpuModuleKeeper
//
// Constancy multithreading: After creation (and before destruction),
// GpuModuleKeeper state is constant, so multi-threaded access is allowed.
//
//================================================================

class GpuModuleKeeper
{

public:

    KIT_COMBINE4(CreateKit, ErrorLogKit, ErrorLogExKit, GpuInitKit, MallocKit);

    //----------------------------------------------------------------
    //
    // Creation/destruction stage
    //
    //----------------------------------------------------------------

public:

    stdbool create(const GpuContext& context, stdPars(CreateKit));
    void destroy();

    //----------------------------------------------------------------
    //
    // Execution stage
    //
    //----------------------------------------------------------------

public:

    stdbool fetchKernel(const GpuKernelLink& link, GpuKernel& kernel, stdPars(ErrorLogKit)) const;
    stdbool fetchSampler(const GpuSamplerLink& link, GpuSampler& sampler, stdPars(ErrorLogKit)) const;

    //----------------------------------------------------------------
    //
    // State variables, constant on execution stage.
    //
    //----------------------------------------------------------------

private:

    bool loaded = false;

    // Map module reference index -> internal kernel index
    // Map module reference index -> internal sampler index
    ArrayMemory<Space> modrefToKernelIndex;
    ArrayMemory<Space> modrefToSamplerIndex;

    ArrayObjectMemory<ModuleInfo> moduleInfo;

    ArrayObjectMemory<GpuKernel> kernelHandle; // separate array, for better performance
    ArrayObjectMemory<KernelInfo> kernelInfo;

    ArrayObjectMemory<GpuSampler> samplerHandle; // separate array, for better performance
    ArrayObjectMemory<SamplerInfo> samplerInfo;

};
