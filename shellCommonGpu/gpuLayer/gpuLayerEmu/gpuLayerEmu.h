#pragma once

#if HEXLIB_PLATFORM == 0

#include "gpuLayer/gpuLayer.h"
#include "errorLog/errorLogKit.h"
#include "userOutput/msgLogExKit.h"
#include "gpuLayer/gpuLayerEmu/emuMultiProc.h"
#include "allocation/mallocAllocator/mallocAllocator.h"
#include "allocation/mallocKit.h"

//================================================================
//
// EmuMemoryAllocator
//
// Ignore GPU context, pass to the malloc thunk.
//
//================================================================

template <typename AddrU>
class EmuMemoryAllocator : public GpuMemoryAllocator<AddrU>
{

public:

    void alloc(const GpuContext& context, AddrU size, AddrU alignment, GpuMemoryOwner& owner, AddrU& result, stdParsNull)
        {base.alloc(size, alignment, owner, result, stdPassNullThru);}

    inline EmuMemoryAllocator(const ErrorLogKit& kit)
        : base(kit) {}

private:

    MallocAllocator<AddrU> base;

};

//================================================================
//
// EmuInitApiToolkit
// EmuExecApiToolkit
//
//================================================================

using EmuInitApiToolkit = KitCombine<ErrorLogKit, MsgLogExKit, MallocKit>;
using EmuExecApiToolkit = KitCombine<ErrorLogKit, MsgLogExKit>;

//================================================================
//
// EmuInitApiThunk
//
//================================================================

class EmuInitApiThunk : public GpuInitApi
{

public:

    //
    // Init
    //

    void initialize(stdParsNull);
    void getDeviceCount(int32& deviceCount, stdParsNull);
    void getProperties(int32 deviceIndex, GpuProperties& properties, stdParsNull);

    //
    // Context
    //

    void createContext(int32 deviceIndex, GpuScheduling gpuScheduling, GpuContextOwner& result, void*& baseContext, stdParsNull);
    static void destroyContext(GpuContextDeallocContext& deallocContext);

    using GpuInitApi::threadContextSet;
    void threadContextSet(const GpuContext& context, GpuThreadContextSave& save, stdParsNull) {}
    void threadContextRestore(const GpuThreadContextSave& save, stdParsNull) {}

    //
    // Module
    //

    void createModuleFromBinary(const GpuContext& context, const Array<const uint8>& binary, GpuModuleOwner& result, stdParsNull)
    {
        result.clear();
    }

    //
    // Kernel
    //

    void createKernelFromModule(const GpuModule& module, const char* kernelName, GpuKernelOwner& result, stdParsNull)
    {
        result.clear();
    }

    //
    // Sampler
    //

    void getSamplerFromModule(const GpuModule& module, const char* samplerName, GpuSamplerOwner& result, stdParsNull)
    {
        result.clear();
    }

    //
    // Allocators
    //

    GpuMemoryAllocator<CpuAddrU>& cpuAllocator() {return cpuAllocatorImpl;}
    GpuMemoryAllocator<GpuAddrU>& gpuAllocator() {return gpuAllocatorImpl;}

    //
    // Texture allocation
    //

    int32 textureAllocCount = 0;

    void createTexture(const GpuContext& context, const Point<Space>& size, GpuChannelType chanType, int rank, GpuTextureOwner& result, stdParsNull);
    static void destroyTexture(GpuTextureDeallocContext& deallocContext);

    //
    // Stream creation
    //

    void createStream(const GpuContext& context, bool nullStream, GpuStreamOwner& result, stdParsNull);
    static void destroyStream(GpuStreamDeallocContext& deallocContext);

    //
    // Event creation
    //

    void eventCreate(const GpuContext& context, bool timingEnabled, GpuEventOwner& result, stdParsNull)
    {
        result.clear();
    }

    //
    // Impl thunk part
    //

    inline EmuInitApiThunk(EmuInitApiToolkit kit)
        : kit(kit), cpuAllocatorImpl(kit), gpuAllocatorImpl(kit) {}

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

    EmuInitApiToolkit kit;

    EmuMemoryAllocator<CpuAddrU> cpuAllocatorImpl;
    EmuMemoryAllocator<GpuAddrU> gpuAllocatorImpl;

};

//================================================================
//
// EmuExecApiThunk
//
//================================================================

class EmuExecApiThunk : public GpuExecApi
{

public:

    //
    // Transfer: Array
    //

    #define TMP_MACRO(funcName, SrcAddr, DstAddr) \
        \
        void funcName \
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
        void funcName \
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

    void setSamplerArray
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

    void setSamplerImageEx
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
    // Kernel launching
    //

    void callKernel
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
    // Stream sync
    //

    void waitStream(const GpuStream& stream, stdParsNull);

    //
    // Events
    //

    void recordEvent(const GpuEvent& event, const GpuStream& stream, stdParsNull);
    void putEventDependency(const GpuEvent& event, const GpuStream& stream, stdParsNull);
    void waitEvent(const GpuEvent& event, bool& realWaitHappened, stdParsNull);
    void eventElapsedTime(const GpuEvent& event1, const GpuEvent& event2, float32& time, stdParsNull);

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

    inline EmuExecApiThunk(EmuExecApiToolkit kit)
        : kit(kit) {}

public:

    GpuExecKit getKit()
    {
        return kitCombine
        (
            GpuTransferKit(*this),
            GpuSamplerSetupKit(*this),
            GpuKernelCallingKit(*this),
            GpuStreamWaitingKit(*this),
            GpuEventRecordingKit(*this),
            GpuEventWaitingKit(*this),
            GpuBenchmarkingControlKit(*this),
            GpuCountingPhaseProhibitionControlKit(*this)
        );
    }

private:

    GpuEnqueueMode gpuEnqueueMode = GpuEnqueueNormal;
    GpuCoverageMode gpuCoverageMode = GpuCoverageNone;

    EmuExecApiToolkit kit;

};

//----------------------------------------------------------------

#endif
