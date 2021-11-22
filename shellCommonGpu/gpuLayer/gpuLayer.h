#pragma once

#include "gpuAppliedApi/gpuAppliedApi.h"
#include "allocation/flatMemoryAllocator.h"
#include "gpuLayer/gpuLayerKits.h"
#include "gpuLayer/gpuScheduling.h"

//================================================================
//
// GPU full host interface.
//
//================================================================

//================================================================
//
// GpuInitialization
//
//================================================================

struct GpuInitialization
{
    virtual stdbool initialize(stdNullPars) =0;
    virtual stdbool getDeviceCount(int32& deviceCount, stdNullPars) =0;
    virtual stdbool getProperties(int32 deviceIndex, GpuProperties& properties, stdNullPars) =0;
};

//================================================================
//
// GpuContextCreation
//
//================================================================

using GpuContextDeallocContext = DeallocContext<8, 0xBBE3292F>;

//----------------------------------------------------------------

struct GpuContextOwner : public GpuContext
{
    inline void clear() {owner.clear();}
    ResourceOwner<GpuContextDeallocContext> owner;
};

//----------------------------------------------------------------

struct GpuContextCreation
{
    virtual stdbool createContext(int32 deviceIndex, GpuScheduling gpuScheduling, GpuContextOwner& result, void*& baseContext, stdNullPars) =0;
    virtual stdbool setThreadContext(const GpuContext& context, stdNullPars) =0;
};

//================================================================
//
// GpuModule
//
//================================================================

struct GpuModule : public OpaqueStruct<8> {};

//================================================================
//
// GpuModuleCreation
//
//================================================================

using GpuModuleDeallocContext = DeallocContext<8, 0xAE2EDBDE>;

//----------------------------------------------------------------

struct GpuModuleOwner : public GpuModule
{
    inline void clear() {owner.clear();}
    ResourceOwner<GpuModuleDeallocContext> owner;
};

//----------------------------------------------------------------

struct GpuModuleCreation
{
    virtual stdbool createModuleFromBinary(const GpuContext& context, const Array<const uint8>& binary, GpuModuleOwner& result, stdNullPars) =0;
};

//================================================================
//
// GpuKernel
//
//================================================================

struct GpuKernel : public OpaqueStruct<8> {};

//================================================================
//
// GpuKernelLoading
//
//================================================================

using GpuKernelDeallocContext = DeallocContext<8, 0x2DF01161>;

//----------------------------------------------------------------

struct GpuKernelOwner : public GpuKernel
{
    inline void clear() {owner.clear();}
    ResourceOwner<GpuKernelDeallocContext> owner;
};

//----------------------------------------------------------------

struct GpuKernelLoading
{
    virtual stdbool createKernelFromModule(const GpuModule& module, const char* kernelName, GpuKernelOwner& result, stdNullPars) =0;
};

//================================================================
//
// GpuSampler
//
//================================================================

struct GpuSampler : public OpaqueStruct<8> {};

//================================================================
//
// GpuSamplerLoading
//
//================================================================

using GpuSamplerDeallocContext = DeallocContext<8, 0x8E62DB13>;

//----------------------------------------------------------------

struct GpuSamplerOwner : public GpuSampler
{
    inline void clear() {owner.clear();}
    ResourceOwner<GpuSamplerDeallocContext> owner;
};

//----------------------------------------------------------------

struct GpuSamplerLoading
{
    virtual stdbool getSamplerFromModule(const GpuModule& module, const char* samplerName, GpuSamplerOwner& result, stdNullPars) =0;
};

//================================================================
//
// GpuMemoryAllocator
//
// Gpu allocator interface: allocates in the specified context.
//
// GpuMemoryAllocation
//
//================================================================

using GpuMemoryOwner = MemoryOwner;

//----------------------------------------------------------------

template <typename AddrU>
struct GpuMemoryAllocator
{
    virtual stdbool alloc(const GpuContext& context, AddrU size, AddrU alignment, GpuMemoryOwner& owner, AddrU& result, stdNullPars) =0;
};

//----------------------------------------------------------------

struct GpuMemoryAllocation
{
    virtual GpuMemoryAllocator<CpuAddrU>& cpuAllocator() =0;
    virtual GpuMemoryAllocator<GpuAddrU>& gpuAllocator() =0;
};

//================================================================
//
// GpuStreamCreation
//
//================================================================

using GpuStreamDeallocContext = DeallocContext<8, 0x31209B08>;

//----------------------------------------------------------------

struct GpuStreamOwner : public GpuStream
{
    inline void clear() {owner.clear();}
    ResourceOwner<GpuStreamDeallocContext> owner;
};

//----------------------------------------------------------------

struct GpuStreamCreation
{
    virtual stdbool createStream(const GpuContext& context, bool nullStream, GpuStreamOwner& result, void*& baseStream, stdNullPars) =0;
};

//================================================================
//
// GpuBenchmarkingControl
//
// Interface for benchmarking purposes only.
//
//================================================================

enum GpuEnqueueMode {GpuEnqueueNormal, GpuEnqueueSkip, GpuEnqueueEmptyKernel};
enum GpuCoverageMode {GpuCoverageNone, GpuCoverageActive};

//----------------------------------------------------------------

struct GpuBenchmarkingControl
{
    virtual void setEnqueueMode(GpuEnqueueMode gpuEnqueueMode) =0;
    virtual GpuEnqueueMode getEnqueueMode() =0;

    virtual void setCoverageMode(GpuCoverageMode gpuCoverageMode) =0;
    virtual GpuCoverageMode getCoverageMode() =0;
};

//================================================================
//
// GpuInitApi
//
//================================================================

struct GpuInitApi
    :
    public GpuInitialization,
    public GpuContextCreation,
    public GpuModuleCreation,
    public GpuKernelLoading,
    public GpuSamplerLoading,
    public GpuMemoryAllocation,
    public GpuTextureAllocator,
    public GpuStreamCreation,
    public GpuEventAllocator
{
};

//================================================================
//
// GpuExecApi
//
//================================================================

struct GpuExecApi
    :
    public GpuTransfer,
    public GpuSamplerSetup,
    public GpuKernelCalling,
    public GpuStreamWaiting,
    public GpuEventRecording,
    public GpuEventWaiting,
    public GpuBenchmarkingControl
{
};
