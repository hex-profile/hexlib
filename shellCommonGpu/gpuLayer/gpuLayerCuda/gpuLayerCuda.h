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

KIT_COMBINE2(CudaInitApiThunkKit, DiagnosticKit, MallocKit);

//================================================================
//
// CudaCpuAllocThunk
//
//================================================================

class CudaCpuAllocThunk : public GpuMemoryAllocator<CpuAddrU>
{

public:

    bool alloc(const GpuContext& context, CpuAddrU size, CpuAddrU alignment, GpuMemoryOwner& owner, CpuAddrU& result, stdNullPars);
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

    bool alloc(const GpuContext& context, GpuAddrU size, GpuAddrU alignment, GpuMemoryOwner& owner, GpuAddrU& result, stdNullPars);
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

    bool initialize(stdNullPars);
    bool getDeviceCount(int32& deviceCount, stdNullPars);
    bool getProperties(int32 deviceIndex, GpuProperties& properties, stdNullPars);

    //
    // Context
    //

    bool createContext(int32 deviceIndex, GpuContextOwner& result, void*& baseContext, stdNullPars);
    static void destroyContext(GpuContextDeallocContext& deallocContext);

    bool setThreadContext(const GpuContext& context, stdNullPars);

    //
    // Module
    //

    bool createModuleFromBinary(const GpuContext& context, const Array<const uint8>& binary, GpuModuleOwner& result, stdNullPars);
    static void destroyModule(GpuModuleDeallocContext& deallocContext);

    //
    // Kernel
    //

    bool createKernelFromModule(const GpuModule& module, const char* kernelName, GpuKernelOwner& result, stdNullPars);

    //
    // Sampler
    //

    bool getSamplerFromModule(const GpuModule& module, const char* samplerName, GpuSamplerOwner& result, stdNullPars);

    //
    // Memory allocation
    //

    GpuMemoryAllocator<CpuAddrU>& cpuAllocator() {return cpuAllocatorImpl;}
    GpuMemoryAllocator<GpuAddrU>& gpuAllocator() {return gpuAllocatorImpl;}

    //
    // Texture allocation
    //

    int32 textureAllocCount = 0;

    bool createTexture(const GpuContext& context, const Point<Space>& size, GpuChannelType chanType, int rank, GpuTextureOwner& result, stdNullPars);
    static void destroyTexture(GpuTextureDeallocContext& deallocContext);

    //
    // Stream
    //

    bool createStream(const GpuContext& context, bool nullStream, GpuStreamOwner& result, void*& baseStream, stdNullPars);
    static void destroyStream(GpuStreamDeallocContext& deallocContext);

    //
    // Total profiling coverage.
    //

    bool coverageInit(const GpuStream& stream, Space coverageQueueCapacity, stdNullPars);
    void coverageDeinit(const GpuStream& stream);

    bool coverageGetSyncFlag(const GpuStream& stream);
    void coverageClearSyncFlag(const GpuStream& stream);

    //
    // Event
    //

    bool createEvent(const GpuContext& context, bool timingEnabled, GpuEventOwner& result, stdNullPars);
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
            GpuInitializationKit(*this, 0),
            GpuContextCreationKit(*this, 0),
            GpuModuleCreationKit(*this, 0),
            GpuKernelLoadingKit(*this, 0),
            GpuSamplerLoadingKit(*this, 0),
            GpuMemoryAllocationKit(*this, 0),
            GpuTextureAllocKit(*this, 0),
            GpuStreamCreationKit(*this, 0),
            GpuEventAllocKit(*this, 0)
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

KIT_COMBINE3(CudaExecApiThunkKit, DiagnosticKit, TimerKit, ProfilerKit);

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
        bool funcName \
        ( \
            SrcAddr srcAddr, \
            DstAddr dstAddr, \
            Space size, \
            const GpuStream& stream, \
            stdNullPars \
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
        bool funcName \
        ( \
            SrcAddr srcAddr, Space srcBytePitch, \
            DstAddr dstAddr, Space dstBytePitch, \
            Space byteSizeX, Space sizeY, \
            const GpuStream& stream, \
            stdNullPars \
        );

    TMP_MACRO(copyMatrixCpuCpu, CpuAddrU, CpuAddrU)
    TMP_MACRO(copyMatrixCpuGpu, CpuAddrU, GpuAddrU)
    TMP_MACRO(copyMatrixGpuCpu, GpuAddrU, CpuAddrU)
    TMP_MACRO(copyMatrixGpuGpu, GpuAddrU, GpuAddrU)

    #undef TMP_MACRO

    //
    // Sampler setup
    //

    bool setSamplerArray
    (
        const GpuSamplerLink& sampler,
        GpuAddrU arrayAddr,
        Space arrayByteSize,
        GpuChannelType chanType,
        int rank,
        BorderMode borderMode,
        bool linearInterpolation,
        bool readNormalizedFloat,
        bool normalizedCoords,
        const GpuContext& context,
        stdNullPars
    );

    bool setSamplerImage
    (
        const GpuSamplerLink& sampler,
        GpuAddrU imageBaseAddr,
        Space imageBytePitch,
        const Point<Space>& imageSize,
        GpuChannelType chanType,
        int rank,
        BorderMode borderMode,
        bool linearInterpolation,
        bool readNormalizedFloat,
        bool normalizedCoords,
        const GpuContext& context,
        stdNullPars
    );

    //
    // Kernel calling
    //

    bool callKernel
    (
        const Point3D<Space>& groupCount,
        const Point<Space>& threadCount,
        uint32 dbgElemCount,
        const GpuKernelLink& kernelLink,
        const void* paramPtr, size_t paramSize,
        const GpuStream& stream,
        stdNullPars
    );

    //
    // Sync
    //

    bool waitStream(const GpuStream& stream, stdNullPars);

    //
    // Events
    //

    bool putEvent(const GpuEvent& event, const GpuStream& stream, stdNullPars);
    bool putEventDependency(const GpuEvent& event, const GpuStream& stream, stdNullPars);

    bool checkEvent(const GpuEvent& event, stdNullPars);
    bool waitEvent(const GpuEvent& event, bool& realWaitHappened, stdNullPars);
    bool eventElapsedTime(const GpuEvent& event1, const GpuEvent& event2, float32& time, stdNullPars);

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
            GpuTransferKit(*this, 0),
            GpuSamplerSetupKit(*this, 0),
            GpuKernelCallingKit(*this, 0),
            GpuStreamWaitingKit(*this, 0),
            GpuEventRecordingKit(*this, 0),
            GpuEventWaitingKit(*this, 0),
            GpuBenchmarkingControlKit(*this, 0)
        );
    }

private:

    CudaExecApiThunkKit kit;

    GpuEnqueueMode gpuEnqueueMode = GpuEnqueueNormal;
    GpuCoverageMode gpuCoverageMode = GpuCoverageNone;

};
