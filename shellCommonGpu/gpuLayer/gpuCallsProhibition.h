#pragma once

#include "userOutput/errorLogEx.h"
#include "errorLog/errorLog.h"
#include "gpuAppliedApi/gpuAppliedApi.h"
#include "gpuLayer/gpuLayer.h"

//================================================================
//
// GPU_PROHIBITED_API_CALL
//
//================================================================

#define GPU_PROHIBITED_API_CALL \
    stdBegin; \
    REQUIRE_TRACE0(false, STR("Prohibited GPU API call in memory counting phase")); \
    stdEnd

//================================================================
//
// GpuProhibitedExecApiThunk
//
//================================================================

class GpuProhibitedExecApiThunk : public GpuExecApi
{

public:

    //----------------------------------------------------------------
    //
    // Transfers
    //
    //----------------------------------------------------------------

    //
    // Array
    //

    #define TMP_MACRO(funcName, SrcAddr, DstAddr) \
        \
        virtual bool funcName \
        ( \
            SrcAddr srcAddr, \
            DstAddr dstAddr, \
            Space size, \
            const GpuStream& stream, \
            stdNullPars \
        ) \
        { \
            GPU_PROHIBITED_API_CALL; \
        }

    TMP_MACRO(copyArrayCpuCpu, CpuAddrU, CpuAddrU)
    TMP_MACRO(copyArrayCpuGpu, CpuAddrU, GpuAddrU)
    TMP_MACRO(copyArrayGpuCpu, GpuAddrU, CpuAddrU)
    TMP_MACRO(copyArrayGpuGpu, GpuAddrU, GpuAddrU)

    #undef TMP_MACRO

    //
    // Matrix
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
        ) \
        { \
            GPU_PROHIBITED_API_CALL; \
        }

    TMP_MACRO(copyMatrixCpuCpu, CpuAddrU, CpuAddrU)
    TMP_MACRO(copyMatrixCpuGpu, CpuAddrU, GpuAddrU)
    TMP_MACRO(copyMatrixGpuCpu, GpuAddrU, CpuAddrU)
    TMP_MACRO(copyMatrixGpuGpu, GpuAddrU, GpuAddrU)

    #undef TMP_MACRO

    //----------------------------------------------------------------
    //
    // Sampler setup
    //
    //----------------------------------------------------------------

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
    )
    {
        GPU_PROHIBITED_API_CALL;
    }

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
    )
    {
        GPU_PROHIBITED_API_CALL;
    }

    //----------------------------------------------------------------
    //
    // Kernel calling
    //
    //----------------------------------------------------------------

    bool callKernel
    (
        const Point3D<Space>& groupCount,
        const Point<Space>& threadCount,
        uint32 dbgElemCount,
        const GpuKernelLink& kernelLink,
        const void* paramPtr, size_t paramSize,
        const GpuStream& stream,
        stdNullPars
    )
    {
        GPU_PROHIBITED_API_CALL;
    }

    //----------------------------------------------------------------
    //
    // Stream sync
    //
    //----------------------------------------------------------------

    bool waitStream(const GpuStream& stream, stdNullPars)
    {
        GPU_PROHIBITED_API_CALL;
    }

    //----------------------------------------------------------------
    //
    // Event
    //
    //----------------------------------------------------------------

    bool putEvent(const GpuEvent& event, const GpuStream& stream, stdNullPars)
    {
        GPU_PROHIBITED_API_CALL;
    }

    bool putEventDependency(const GpuEvent& event, const GpuStream& stream, stdNullPars)
    {
        GPU_PROHIBITED_API_CALL;
    }

    bool checkEvent(const GpuEvent& event, stdNullPars)
    {
        GPU_PROHIBITED_API_CALL;
    }

    bool waitEvent(const GpuEvent& event, bool& realWaitHappened, stdNullPars)
    {
        GPU_PROHIBITED_API_CALL;
    }

    bool eventElapsedTime(const GpuEvent& event1, const GpuEvent& event2, float32& time, stdNullPars)
    {
        GPU_PROHIBITED_API_CALL;
    }

    //----------------------------------------------------------------
    //
    // Benchmarking control
    //
    //----------------------------------------------------------------

    void setEnqueueMode(GpuEnqueueMode gpuEnqueueMode)
        {}

    GpuEnqueueMode getEnqueueMode()
        {return GpuEnqueueNormal;}

    void setCoverageMode(GpuCoverageMode gpuCoverageMode)
        {}

    GpuCoverageMode getCoverageMode()
        {return GpuCoverageNone;}

    //----------------------------------------------------------------
    //
    //
    //
    //----------------------------------------------------------------

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

    //----------------------------------------------------------------
    //
    // Impl
    //
    //----------------------------------------------------------------

    inline GpuProhibitedExecApiThunk(const ErrorLogExKit& kit)
        : kit(kit) {}

private:

    ErrorLogExKit kit;

};

//================================================================
//
// GpuEventAllocatorSupressor
//
//================================================================

class GpuEventAllocatorSupressor : public GpuEventAllocator
{

public:

    bool createEvent(const GpuContext& context, bool timingEnabled, GpuEventOwner& result, stdNullPars)
    {
        result.clear();
        return true;
    }

    GpuEventAllocatorSupressor(const ErrorLogExKit& kit)
        : kit(kit) {}

private:

    ErrorLogExKit kit;

};

//================================================================
//
// GpuTextureAllocatorSupressor
//
//================================================================

class GpuTextureAllocatorSupressor : public GpuTextureAllocator
{

public:

    bool createTexture(const GpuContext& context, const Point<Space>& size, GpuChannelType chanType, int rank, GpuTextureOwner& result, stdNullPars)
    {
        result.clear();
        return true;
    }

    GpuTextureAllocatorSupressor(const ErrorLogExKit& kit)
        : kit(kit) {}

private:

    ErrorLogExKit kit;

};
