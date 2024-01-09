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

    stdbool alloc(const GpuContext& context, AddrU size, AddrU alignment, GpuMemoryOwner& owner, AddrU& result, stdParsNull)
        {return base.alloc(size, alignment, owner, result, stdPassNullThru);}

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

    stdbool initialize(stdParsNull);
    stdbool getDeviceCount(int32& deviceCount, stdParsNull);
    stdbool getProperties(int32 deviceIndex, GpuProperties& properties, stdParsNull);

    //
    // Context
    //

    stdbool createContext(int32 deviceIndex, GpuScheduling gpuScheduling, GpuContextOwner& result, void*& baseContext, stdParsNull);
    static void destroyContext(GpuContextDeallocContext& deallocContext);

    using GpuInitApi::threadContextSet;
    stdbool threadContextSet(const GpuContext& context, GpuThreadContextSave& save, stdParsNull) {returnTrue;}
    stdbool threadContextRestore(const GpuThreadContextSave& save, stdParsNull) {returnTrue;}

    //
    // Module
    //

    stdbool createModuleFromBinary(const GpuContext& context, const Array<const uint8>& binary, GpuModuleOwner& result, stdParsNull)
    {
        result.clear();
        returnTrue;
    }

    //
    // Kernel
    //

    stdbool createKernelFromModule(const GpuModule& module, const char* kernelName, GpuKernelOwner& result, stdParsNull)
    {
        result.clear();
        returnTrue;
    }

    //
    // Sampler
    //

    stdbool getSamplerFromModule(const GpuModule& module, const char* samplerName, GpuSamplerOwner& result, stdParsNull)
    {
        result.clear();
        returnTrue;
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

    stdbool createTexture(const GpuContext& context, const Point<Space>& size, GpuChannelType chanType, int rank, GpuTextureOwner& result, stdParsNull);
    static void destroyTexture(GpuTextureDeallocContext& deallocContext);

    //
    // Stream creation
    //

    stdbool createStream(const GpuContext& context, bool nullStream, GpuStreamOwner& result, stdParsNull);
    static void destroyStream(GpuStreamDeallocContext& deallocContext);

    //
    // Event creation
    //

    stdbool eventCreate(const GpuContext& context, bool timingEnabled, GpuEventOwner& result, stdParsNull)
    {
        result.clear();
        returnTrue;
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
    // Kernel launching
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
    // Stream sync
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
