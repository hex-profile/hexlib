#pragma once

#include "gpuAppliedApi/gpuAppliedApi.h"
#include "gpuLayer/gpuLayerKits.h"
#include "gpuLayer/gpuScheduling.h"
#include "dataAlloc/memoryAllocator.h"

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
    virtual void initialize(stdParsNull) =0;
    virtual void getDeviceCount(int32& deviceCount, stdParsNull) =0;
    virtual void getProperties(int32 deviceIndex, GpuProperties& properties, stdParsNull) =0;
};

//================================================================
//
// GpuContextCreation
//
//================================================================

using GpuContextDeallocContext = OpaqueStruct<8, 0xBBE3292Fu>;

//----------------------------------------------------------------

struct GpuContextOwner : public GpuContext
{
    inline void clear() {owner.clear();}
    ResourceOwner<GpuContextDeallocContext> owner;
};

//----------------------------------------------------------------

struct GpuContextCreation
{
    virtual void createContext(int32 deviceIndex, GpuScheduling gpuScheduling, GpuContextOwner& result, void*& baseContext, stdParsNull) =0;
};

//================================================================
//
// GpuContextSetting
//
//================================================================

using GpuThreadContextSave = OpaqueStruct<8, 0xFD45FC43u>;

//----------------------------------------------------------------

struct GpuContextSetting
{
    virtual void threadContextSet(const GpuContext& context, GpuThreadContextSave& save, stdParsNull) =0;
    virtual void threadContextRestore(const GpuThreadContextSave& save, stdParsNull) =0;

    inline void threadContextSet(const GpuContext& context, stdParsNull)
    {
        GpuThreadContextSave tmp;
        threadContextSet(context, tmp, stdPassNull);
    }
};

//================================================================
//
// GpuModule
//
//================================================================

using GpuModule = OpaqueStruct<8, 0xB6F8F12Fu>;

//================================================================
//
// GpuModuleCreation
//
//================================================================

using GpuModuleDeallocContext = OpaqueStruct<8, 0xAE2EDBDEu>;

//----------------------------------------------------------------

struct GpuModuleOwner : public GpuModule
{
    inline void clear() {owner.clear();}
    ResourceOwner<GpuModuleDeallocContext> owner;
};

//----------------------------------------------------------------

struct GpuModuleCreation
{
    virtual void createModuleFromBinary(const GpuContext& context, const Array<const uint8>& binary, GpuModuleOwner& result, stdParsNull) =0;
};

//================================================================
//
// GpuKernel
//
//================================================================

using GpuKernel = OpaqueStruct<8, 0x5A486D88u>;

//================================================================
//
// GpuKernelLoading
//
//================================================================

using GpuKernelDeallocContext = OpaqueStruct<8, 0x2DF01161u>;

//----------------------------------------------------------------

struct GpuKernelOwner : public GpuKernel
{
    inline void clear() {owner.clear();}
    ResourceOwner<GpuKernelDeallocContext> owner;
};

//----------------------------------------------------------------

struct GpuKernelLoading
{
    virtual void createKernelFromModule(const GpuModule& module, const char* kernelName, GpuKernelOwner& result, stdParsNull) =0;
};

//================================================================
//
// GpuSampler
//
//================================================================

using GpuSampler = OpaqueStruct<8, 0x729E550Fu>;

//================================================================
//
// GpuSamplerLoading
//
//================================================================

using GpuSamplerDeallocContext = OpaqueStruct<8, 0x8E62DB13u>;

//----------------------------------------------------------------

struct GpuSamplerOwner : public GpuSampler
{
    inline void clear() {owner.clear();}
    ResourceOwner<GpuSamplerDeallocContext> owner;
};

//----------------------------------------------------------------

struct GpuSamplerLoading
{
    virtual void getSamplerFromModule(const GpuModule& module, const char* samplerName, GpuSamplerOwner& result, stdParsNull) =0;
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
    virtual void alloc(const GpuContext& context, AddrU size, AddrU alignment, GpuMemoryOwner& owner, AddrU& result, stdParsNull) =0;
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

using GpuStreamDeallocContext = OpaqueStruct<8, 0x31209B08u>;

//----------------------------------------------------------------

struct GpuStreamOwner : public GpuStream
{
    inline void clear() {owner.clear();}
    ResourceOwner<GpuStreamDeallocContext> owner;
};

//----------------------------------------------------------------

struct GpuStreamCreation
{
    virtual void createStream(const GpuContext& context, bool nullStream, GpuStreamOwner& result, stdParsNull) =0;
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
// GpuCountingPhaseProhibitionControl
//
// This API allows for the temporary suspension of the prohibition on GPU calls
// during memory counting. This ban generally helps to track missed calls,
// which can lead to system slowdown. However, there are instances where it needs
// to be disabled, and this API provides that capability.
//
//================================================================

struct GpuCountingPhaseProhibitionControl
{
    virtual void setCountingPhaseGpuProhibition(bool value) =0;
    virtual bool getCountingPhaseGpuProhibition() =0;
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
    public GpuContextSetting,
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
    public GpuBenchmarkingControl,
    public GpuCountingPhaseProhibitionControl
{
};

//================================================================
//
// getNativeHandle
//
//================================================================

void* getNativeHandle(const GpuStream& stream);
