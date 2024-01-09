#pragma once

#include "gpuLayer/gpuLayer.h"
#include "errorLog/errorLogKit.h"
#include "userOutput/diagnosticKit.h"
#include "gpuLayer/gpuLayerCuda/gpuModuleKeeper.h"
#include "storage/rememberCleanup.h"
#include "timer/timerKit.h"
#include "brood.h"
#include "allocation/mallocKit.h"

//================================================================
//
// CudaInitApiThunkKit
//
//================================================================

using CudaInitApiThunkKit = KitCombine<DiagnosticKit, MallocKit>;

//================================================================
//
// CudaCpuAllocThunk
//
//================================================================

class CudaCpuAllocThunk : public GpuMemoryAllocator<CpuAddrU>
{

public:

    stdbool alloc(const GpuContext& context, CpuAddrU size, CpuAddrU alignment, GpuMemoryOwner& owner, CpuAddrU& result, stdParsNull);
    static void dealloc(MemoryDeallocContext& deallocContext);

    CudaCpuAllocThunk(const CudaInitApiThunkKit& kit) : kit(kit) {}

private:

    CudaInitApiThunkKit kit;

};

//================================================================
//
// CudaGpuAllocThunk
//
//================================================================

class CudaGpuAllocThunk : public GpuMemoryAllocator<GpuAddrU>
{

public:

    stdbool alloc(const GpuContext& context, GpuAddrU size, GpuAddrU alignment, GpuMemoryOwner& owner, GpuAddrU& result, stdParsNull);
    static void dealloc(MemoryDeallocContext& deallocContext);

    CudaGpuAllocThunk(const CudaInitApiThunkKit& kit) : kit(kit) {}

private:

    CudaInitApiThunkKit kit;

};

//================================================================
//
// CudaInitApiThunk
//
//================================================================

class CudaInitApiThunk : public GpuInitApi
{

public:

    //
    // Init
    //

    stdbool initialize(stdParsNull);
    stdbool getDeviceCount(int32& deviceCount, stdParsNull);
    stdbool getProperties(int32 deviceIndex, GpuProperties& properties, stdParsNull);

    //
    // Context
    //

    stdbool createContext(int32 deviceIndex, GpuScheduling gpuScheduling, GpuContextOwner& result, void*& baseContext, stdParsNull);
    static void destroyContext(GpuContextDeallocContext& deallocContext);

    using GpuInitApi::threadContextSet;
    stdbool threadContextSet(const GpuContext& context, GpuThreadContextSave& save, stdParsNull);
    stdbool threadContextRestore(const GpuThreadContextSave& save, stdParsNull);

    //
    // Module
    //

    stdbool createModuleFromBinary(const GpuContext& context, const Array<const uint8>& binary, GpuModuleOwner& result, stdParsNull);
    static void destroyModule(GpuModuleDeallocContext& deallocContext);

    //
    // Kernel
    //

    stdbool createKernelFromModule(const GpuModule& module, const char* kernelName, GpuKernelOwner& result, stdParsNull);

    //
    // Sampler
    //

    stdbool getSamplerFromModule(const GpuModule& module, const char* samplerName, GpuSamplerOwner& result, stdParsNull);

    //
    // Memory allocation
    //

    GpuMemoryAllocator<CpuAddrU>& cpuAllocator() {return cpuAllocatorImpl;}
    GpuMemoryAllocator<GpuAddrU>& gpuAllocator() {return gpuAllocatorImpl;}

    //
    // Texture allocation
    //

    int32 textureAllocCount = 0;

    stdbool createTexture(const GpuContext& context, const Point<Space>& size, GpuChannelType chanType, int rank, GpuTextureOwner& result, stdParsNull);
    static void destroyTexture(GpuTextureDeallocContext& deallocContext);

    //
    // Stream
    //

    stdbool createStream(const GpuContext& context, bool nullStream, GpuStreamOwner& result, stdParsNull);
    static void destroyStream(GpuStreamDeallocContext& deallocContext);

    //
    // Total profiling coverage.
    //

    stdbool coverageInit(const GpuStream& stream, Space coverageQueueCapacity, stdParsNull);
    void coverageDeinit(const GpuStream& stream);

    bool coverageGetSyncFlag(const GpuStream& stream);
    void coverageClearSyncFlag(const GpuStream& stream);

    //
    // Event
    //

    stdbool eventCreate(const GpuContext& context, bool timingEnabled, GpuEventOwner& result, stdParsNull);
    static void destroyEvent(GpuEventDeallocContext& deallocContext);

    //
    // Impl thunk part
    //

public:

    inline CudaInitApiThunk(CudaInitApiThunkKit kit)
        :
        kit(kit),
        cpuAllocatorImpl(kit),
        gpuAllocatorImpl(kit)
    {
    }

public:

    inline GpuInitKit getKit()
    {
        return kitCombine
        (
            GpuInitializationKit(*this),
            GpuContextCreationKit(*this),
            GpuContextSettingKit(*this),
            GpuModuleCreationKit(*this),
            GpuKernelLoadingKit(*this),
            GpuSamplerLoadingKit(*this),
            GpuMemoryAllocationKit(*this),
            GpuTextureAllocKit(*this),
            GpuStreamCreationKit(*this),
            GpuEventAllocKit(*this)
        );
    }

private:

    CudaInitApiThunkKit kit;

    CudaCpuAllocThunk cpuAllocatorImpl;
    CudaGpuAllocThunk gpuAllocatorImpl;

};

//================================================================
//
// CudaExecApiThunkKit
//
//================================================================

using CudaExecApiThunkKit = KitCombine<DiagnosticKit, TimerKit, ProfilerKit>;

//================================================================
//
// CudaExecApiThunk
//
//================================================================

class CudaExecApiThunk : public GpuExecApi
{

public:

    //
    // Transfer: Array
    //

    #define TMP_MACRO(funcName, SrcAddr, DstAddr) \
        \
        stdbool funcName \
        ( \
            SrcAddr srcAddr, \
            DstAddr dstAddr, \
            Space size, \
            const GpuStream& stream, \
            stdParsNull \
        );

    TMP_MACRO(copyArrayCpuCpu, CpuAddrU, CpuAddrU)
    TMP_MACRO(copyArrayCpuGpu, CpuAddrU, GpuAddrU)
    TMP_MACRO(copyArrayGpuCpu, GpuAddrU, CpuAddrU)
    TMP_MACRO(copyArrayGpuGpu, GpuAddrU, GpuAddrU)

    #undef TMP_MACRO

    //
    // Transfer: Matrix
    //

    #define TMP_MACRO(funcName, SrcAddr, DstAddr) \
        \
        stdbool funcName \
        ( \
            SrcAddr srcAddr, Space srcBytePitch, \
            DstAddr dstAddr, Space dstBytePitch, \
            Space byteSizeX, Space sizeY, \
            const GpuStream& stream, \
            stdParsNull \
        );

    TMP_MACRO(copyMatrixCpuCpu, CpuAddrU, CpuAddrU)
    TMP_MACRO(copyMatrixCpuGpu, CpuAddrU, GpuAddrU)
    TMP_MACRO(copyMatrixGpuCpu, GpuAddrU, CpuAddrU)
    TMP_MACRO(copyMatrixGpuGpu, GpuAddrU, GpuAddrU)

    #undef TMP_MACRO

    //
    // Sampler setup
    //

    stdbool setSamplerArray
    (
        const GpuSamplerLink& sampler,
        GpuAddrU arrayAddr,
        Space arrayByteSize,
        GpuChannelType chanType,
        int rank,
        BorderMode borderMode,
        LinearInterpolation linearInterpolation,
        ReadNormalizedFloat readNormalizedFloat,
        NormalizedCoords normalizedCoords,
        const GpuContext& context,
        stdParsNull
    );

    stdbool setSamplerImageEx
    (
        const GpuSamplerLink& sampler,
        GpuAddrU imageBaseAddr,
        Space imageBytePitch,
        const Point<Space>& imageSize,
        GpuChannelType chanType,
        int rank,
        BorderMode borderMode,
        LinearInterpolation linearInterpolation,
        ReadNormalizedFloat readNormalizedFloat,
        NormalizedCoords normalizedCoords,
        const GpuContext& context,
        stdParsNull
    );

    //
    // Kernel calling
    //

    stdbool callKernel
    (
        const Point3D<Space>& groupCount,
        const Point<Space>& threadCount,
        uint32 dbgElemCount,
        const GpuKernelLink& kernelLink,
        const void* paramPtr, size_t paramSize,
        const GpuStream& stream,
        stdParsNull
    );

    //
    // Sync
    //

    stdbool waitStream(const GpuStream& stream, stdParsNull);

    //
    // Events
    //

    stdbool recordEvent(const GpuEvent& event, const GpuStream& stream, stdParsNull);
    stdbool putEventDependency(const GpuEvent& event, const GpuStream& stream, stdParsNull);
    stdbool waitEvent(const GpuEvent& event, bool& realWaitHappened, stdParsNull);
    stdbool eventElapsedTime(const GpuEvent& event1, const GpuEvent& event2, float32& time, stdParsNull);

    //
    // Benchmarking control
    //

    void setEnqueueMode(GpuEnqueueMode gpuEnqueueMode)
        {this->gpuEnqueueMode = gpuEnqueueMode;}

    GpuEnqueueMode getEnqueueMode()
        {return gpuEnqueueMode;}

    void setCoverageMode(GpuCoverageMode gpuCoverageMode)
        {this->gpuCoverageMode = gpuCoverageMode;}

    GpuCoverageMode getCoverageMode()
        {return gpuCoverageMode;}

    //
    // Counting phase GPU prohibition control
    //

    void setCountingPhaseGpuProhibition(bool value)
        {}

    bool getCountingPhaseGpuProhibition()
        {return true;}

    //
    // Impl thunk part
    //

public:

    inline CudaExecApiThunk(CudaExecApiThunkKit kit)
        : kit(kit) {}

public:

    GpuExecKit getKit()
    {
        return kitCombine
        (
            GpuTransferKit{*this},
            GpuSamplerSetupKit{*this},
            GpuKernelCallingKit{*this},
            GpuStreamWaitingKit{*this},
            GpuEventRecordingKit{*this},
            GpuEventWaitingKit{*this},
            GpuBenchmarkingControlKit{*this},
            GpuCountingPhaseProhibitionControlKit{*this}
        );
    }

private:

    CudaExecApiThunkKit kit;

    GpuEnqueueMode gpuEnqueueMode = GpuEnqueueNormal;
    GpuCoverageMode gpuCoverageMode = GpuCoverageNone;

};
