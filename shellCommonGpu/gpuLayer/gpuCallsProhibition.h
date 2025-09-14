#pragma once

#include "userOutput/printMsgTrace.h"
#include "errorLog/errorLog.h"
#include "gpuAppliedApi/gpuAppliedApi.h"
#include "gpuLayer/gpuLayer.h"

//================================================================
//
// GPU_PROHIBITED_API_CALL
//
//================================================================

#define GPU_PROHIBITED_API_CALL \
    REQUIRE_TRACE(false, STR("Prohibited GPU API call in memory counting phase"))

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
        virtual void funcName \
        ( \
            SrcAddr srcAddr, \
            DstAddr dstAddr, \
            Space size, \
            const GpuStream& stream, \
            stdParsNull \
        ) \
        { \
            if (prohibitionEnabled) \
                GPU_PROHIBITED_API_CALL; \
            \
            kit.gpuTransfer.funcName(srcAddr, dstAddr, size, stream, stdPassNullThru); \
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
        void funcName \
        ( \
            SrcAddr srcAddr, Space srcBytePitch, \
            DstAddr dstAddr, Space dstBytePitch, \
            Space byteSizeX, Space sizeY, \
            const GpuStream& stream, \
            stdParsNull \
        ) \
        { \
            if (prohibitionEnabled) \
                GPU_PROHIBITED_API_CALL; \
            \
            kit.gpuTransfer.funcName(srcAddr, srcBytePitch, dstAddr, dstBytePitch, byteSizeX, sizeY, stream, stdPassNullThru); \
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
    )
    {
        if (prohibitionEnabled)
            GPU_PROHIBITED_API_CALL;

        kit.gpuSamplerSetting.setSamplerArray
        (
            sampler,
            arrayAddr,
            arrayByteSize,
            chanType,
            rank,
            borderMode,
            linearInterpolation,
            readNormalizedFloat,
            normalizedCoords,
            context,
            stdPassNullThru
        );
    }

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
    )
    {
        if (prohibitionEnabled)
            GPU_PROHIBITED_API_CALL;

        kit.gpuSamplerSetting.setSamplerImageEx
        (
            sampler,
            imageBaseAddr,
            imageBytePitch,
            imageSize,
            chanType,
            rank,
            borderMode,
            linearInterpolation,
            readNormalizedFloat,
            normalizedCoords,
            context,
            stdPassNullThru
        );
    }

    //----------------------------------------------------------------
    //
    // Kernel calling
    //
    //----------------------------------------------------------------

    void callKernel
    (
        const Point3D<Space>& groupCount,
        const Point<Space>& threadCount,
        uint32 dbgElemCount,
        const GpuKernelLink& kernelLink,
        const void* paramPtr, size_t paramSize,
        const GpuStream& stream,
        stdParsNull
    )
    {
        if (prohibitionEnabled)
            GPU_PROHIBITED_API_CALL;

        kit.gpuKernelCalling.callKernel(groupCount, threadCount, dbgElemCount, kernelLink, paramPtr, paramSize, stream, stdPassNullThru);
    }

    //----------------------------------------------------------------
    //
    // Stream sync
    //
    //----------------------------------------------------------------

    void waitStream(const GpuStream& stream, stdParsNull)
    {
        if (prohibitionEnabled)
            GPU_PROHIBITED_API_CALL;

        kit.gpuStreamWaiting.waitStream(stream, stdPassNullThru);
    }

    //----------------------------------------------------------------
    //
    // Event
    //
    //----------------------------------------------------------------

    void recordEvent(const GpuEvent& event, const GpuStream& stream, stdParsNull)
    {
        if (prohibitionEnabled)
            GPU_PROHIBITED_API_CALL;

        kit.gpuEventRecording.recordEvent(event, stream, stdPassNullThru);
    }

    void putEventDependency(const GpuEvent& event, const GpuStream& stream, stdParsNull)
    {
        if (prohibitionEnabled)
            GPU_PROHIBITED_API_CALL;

        kit.gpuEventRecording.putEventDependency(event, stream, stdPassNullThru);
    }

    void waitEvent(const GpuEvent& event, bool& realWaitHappened, stdParsNull)
    {
        if (prohibitionEnabled)
            GPU_PROHIBITED_API_CALL;

        kit.gpuEventWaiting.waitEvent(event, realWaitHappened, stdPassNullThru);
    }

    void eventElapsedTime(const GpuEvent& event1, const GpuEvent& event2, float32& time, stdParsNull)
    {
        if (prohibitionEnabled)
            GPU_PROHIBITED_API_CALL;

        kit.gpuEventWaiting.eventElapsedTime(event1, event2, time, stdPassNullThru);
    }

    //----------------------------------------------------------------
    //
    // Benchmarking control
    //
    //----------------------------------------------------------------

    void setEnqueueMode(GpuEnqueueMode gpuEnqueueMode)
        {kit.gpuBenchmarkingControl.setEnqueueMode(gpuEnqueueMode);}

    GpuEnqueueMode getEnqueueMode()
        {return kit.gpuBenchmarkingControl.getEnqueueMode();}

    void setCoverageMode(GpuCoverageMode gpuCoverageMode)
        {kit.gpuBenchmarkingControl.setCoverageMode(gpuCoverageMode);}

    GpuCoverageMode getCoverageMode()
        {return kit.gpuBenchmarkingControl.getCoverageMode();}

    //----------------------------------------------------------------
    //
    // Counting phase GPU prohibition control
    //
    //----------------------------------------------------------------

    void setCountingPhaseGpuProhibition(bool value)
        {prohibitionEnabled = value;}

    bool getCountingPhaseGpuProhibition()
        {return prohibitionEnabled;}

    //----------------------------------------------------------------
    //
    // Get kit
    //
    //----------------------------------------------------------------

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

    //----------------------------------------------------------------
    //
    // Impl
    //
    //----------------------------------------------------------------

public:

    using Kit = KitCombine<GpuExecKit, MsgLogExKit>;

    inline GpuProhibitedExecApiThunk(const Kit& kit)
        : kit(kit) {}

private:

    Kit kit;
    bool prohibitionEnabled = true;

};
