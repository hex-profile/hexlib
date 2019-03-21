#pragma once

#include "errorLog/errorLog.h"
#include "gpuLayer/gpuLayer.h"
#include "dataAlloc/arrayObjMem.h"
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

    bool create(const GpuContext& context, stdPars(CreateKit));
    void destroy();

    //----------------------------------------------------------------
    //
    // Execution stage
    //
    //----------------------------------------------------------------

public:

    bool fetchKernel(const GpuKernelLink& link, GpuKernel& kernel, stdPars(ErrorLogKit)) const;
    bool fetchSampler(const GpuSamplerLink& link, GpuSampler& sampler, stdPars(ErrorLogKit)) const;

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

    ArrayObjMem<ModuleInfo> moduleInfo;

    ArrayObjMem<GpuKernel> kernelHandle; // separate array, for better performance
    ArrayObjMem<KernelInfo> kernelInfo;

    ArrayObjMem<GpuSampler> samplerHandle; // separate array, for better performance
    ArrayObjMem<SamplerInfo> samplerInfo;

};
