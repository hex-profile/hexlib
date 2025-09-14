#pragma once

#include "errorLog/errorLog.h"
#include "gpuLayer/gpuLayer.h"
#include "dataAlloc/arrayObjectMemory.h"
#include "dataAlloc/memoryAllocatorKit.h"
#include "userOutput/printMsgTrace.h"
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

    using CreateKit = KitCombine<ErrorLogKit, MsgLogExKit, GpuInitKit, MallocKit>;

    //----------------------------------------------------------------
    //
    // Get statistics.
    //
    //----------------------------------------------------------------

public:

    static void getStatistics(int32& moduleCount, int32& kernelCount, int32& samplerCount, stdPars(ErrorLogKit));

    //----------------------------------------------------------------
    //
    // Creation/destruction stage
    //
    //----------------------------------------------------------------

public:

    void create(const GpuContext& context, stdPars(CreateKit));
    void destroy();

    //----------------------------------------------------------------
    //
    // Execution stage
    //
    //----------------------------------------------------------------

public:

    void fetchKernel(const GpuKernelLink& link, GpuKernel& kernel, stdPars(ErrorLogKit)) const;
    void fetchSampler(const GpuSamplerLink& link, GpuSampler& sampler, stdPars(ErrorLogKit)) const;

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
