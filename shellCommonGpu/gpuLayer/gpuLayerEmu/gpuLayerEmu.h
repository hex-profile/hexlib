#pragma once

#if HEXLIB_PLATFORM == 0

#include "gpuLayer/gpuLayer.h"
#include "errorLog/errorLogKit.h"
#include "userOutput/errorLogExKit.h"
#include "gpuLayer/gpuLayerEmu/emuMultiProc.h"
#include "allocation/mallocFlatAllocator/mallocFlatAllocator.h"
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

    stdbool alloc(const GpuContext& context, AddrU size, AddrU alignment, GpuMemoryOwner& owner, AddrU& result, stdNullPars)
        {return base.alloc(size, alignment, owner, result, stdNullPassThru);}

    inline EmuMemoryAllocator(const ErrorLogKit& kit)
        : base(kit) {}

private:

    MallocFlatAllocatorThunk<AddrU> base;

};

//================================================================
//
// EmuInitApiToolkit
// EmuExecApiToolkit
//
//================================================================

using EmuInitApiToolkit = KitCombine<ErrorLogKit, ErrorLogExKit, MallocKit>;
using EmuExecApiToolkit = KitCombine<ErrorLogKit, ErrorLogExKit>;

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

    stdbool initialize(stdNullPars);
    stdbool getDeviceCount(int32& deviceCount, stdNullPars);
    stdbool getProperties(int32 deviceIndex, GpuProperties& properties, stdNullPars);

    //
    // Context
    //

    stdbool createContext(int32 deviceIndex, GpuContextOwner& result, void*& baseContext, stdNullPars);
    static void destroyContext(GpuContextDeallocContext& deallocContext);
    stdbool setThreadContext(const GpuContext& context, stdNullPars) {returnTrue;}

    //
    // Module
    //

    stdbool createModuleFromBinary(const GpuContext& context, const Array<const uint8>& binary, GpuModuleOwner& result, stdNullPars)
    {
        result.clear();
        returnTrue;
    }

    //
    // Kernel
    //

    stdbool createKernelFromModule(const GpuModule& module, const char* kernelName, GpuKernelOwner& result, stdNullPars)
    {
        result.clear();
        returnTrue;
    }

    //
    // Sampler
    //

    stdbool getSamplerFromModule(const GpuModule& module, const char* samplerName, GpuSamplerOwner& result, stdNullPars)
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

    stdbool createTexture(const GpuContext& context, const Point<Space>& size, GpuChannelType chanType, int rank, GpuTextureOwner& result, stdNullPars);
    static void destroyTexture(GpuTextureDeallocContext& deallocContext);

    //
    // Stream creation
    //

    stdbool createStream(const GpuContext& context, bool nullStream, GpuStreamOwner& result, void*& baseStream, stdNullPars);
    static void destroyStream(GpuStreamDeallocContext& deallocContext);

    //
    // Event creation
    //

    stdbool eventCreate(const GpuContext& context, bool timingEnabled, GpuEventOwner& result, stdNullPars)
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
        stdbool funcName \
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

    stdbool setSamplerArray
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

    stdbool setSamplerImage
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
        stdNullPars
    );

    //
    // Stream sync
    //

    stdbool waitStream(const GpuStream& stream, stdNullPars);

    //
    // Events
    //

    stdbool putEvent(const GpuEvent& event, const GpuStream& stream, stdNullPars);
    stdbool putEventDependency(const GpuEvent& event, const GpuStream& stream, stdNullPars);
    stdbool checkEvent(const GpuEvent& event, stdNullPars);
    stdbool waitEvent(const GpuEvent& event, bool& realWaitHappened, stdNullPars);
    stdbool eventElapsedTime(const GpuEvent& event1, const GpuEvent& event2, float32& time, stdNullPars);

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
            GpuBenchmarkingControlKit(*this)
        );
    }

private:

    GpuEnqueueMode gpuEnqueueMode = GpuEnqueueNormal;
    GpuCoverageMode gpuCoverageMode = GpuCoverageNone;

    EmuExecApiToolkit kit;

};

//----------------------------------------------------------------

#endif
