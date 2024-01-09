#pragma once

#include "gpuAppliedKits.h"

#include "errorLog/errorLog.h"
#include "numbers/float/floatBase.h"
#include "stdFunc/stdFunc.h"
#include "storage/opaqueStruct.h"
#include "data/gpuArray.h"
#include "data/gpuMatrix.h"
#include "data/gpuPtr.h"
#include "gpuAppliedApi/gpuSamplerSetup.h"
#include "dataAlloc/deallocInterface.h"
#include "data/matrix.h"
#include "point3d/point3d.h"

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// GPU applied API represents a minimal required subset of full GPU API
// for using in a mass of application code.
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================


//================================================================
//
// GpuProperties
//
//================================================================

struct GpuProperties
{
    // The number of threads required to saturate the whole GPU.
    int32 occupancyThreadsGood{};
    int32 occupancyThreadsMax{};

    // Total GPU throughput in instructions per second.
    float32 totalThroughput{};

    // Sampler base address alignment in bytes.
    //
    // Also serves as a base alignment for CPU and GPU
    // memory blocks involved in DMA transfers,
    // like CPU pinned memory blocks and their GPU counterparts.
    SpaceU samplerAndFastTransferBaseAlignment{};

    // Sampler row alignment in bytes.
    // This row alignment is >= good DRAM access alignment.
    SpaceU samplerRowAlignment{};

    // Max group count and group size
    Point3D<SpaceU> maxGroupCount{};
    Point<Space> maxThreadCount{};
    Space maxGroupArea{};
};

//================================================================
//
// GpuTextureAllocator
//
//================================================================

using GpuTextureDeallocContext = OpaqueStruct<8, 0xB113EE47u>;

//----------------------------------------------------------------

struct GpuTextureOwner : public GpuTexture
{
    inline void clear() {owner.clear();}
    ResourceOwner<GpuTextureDeallocContext> owner;
};

//----------------------------------------------------------------

struct GpuTextureAllocator
{
    virtual stdbool createTexture(const GpuContext& context, const Point<Space>& size, GpuChannelType chanType, int rank, GpuTextureOwner& result, stdParsNull) =0;

    template <typename Type>
    inline stdbool createTexture(const Point<Space>& size, GpuTextureOwner& result, stdParsNull)
        {return createTexture(size, GpuGetChannelType<Type>::val, VectorTypeRank<Type>::val, result, stdPassNullThru);}
};

//================================================================
//
// GpuStreamWaiting
//
// Wait until the command queue finishes execution.
//
//================================================================

struct GpuStreamWaiting
{
    virtual stdbool waitStream(const GpuStream& stream, stdParsNull) =0;
};

//----------------------------------------------------------------

template <typename Kit>
sysinline stdbool gpuSyncCurrentStream(stdPars(Kit))
{
    if (kit.dataProcessing)
        require(kit.gpuStreamWaiting.waitStream(kit.gpuCurrentStream, stdPassThru));

    returnTrue;
}

//================================================================
//
// GpuEventAllocator
//
// An event captures a set of work.
//
// After event creation, the captured work set is EMPTY,
// so waiting on such event returns immediately.
//
//================================================================

using GpuEventDeallocContext = OpaqueStruct<8, 0xF21F8275u>;

//----------------------------------------------------------------

struct GpuEventOwner : public GpuEvent
{
    using Base = GpuEvent;

    inline void clear() {owner.clear();}
    ResourceOwner<GpuEventDeallocContext> owner;

    friend inline void exchange(GpuEventOwner& a, GpuEventOwner& b)
    {
        exchange(soft_cast<Base&>(a), soft_cast<Base&>(b));
        exchange(a.owner, b.owner);
    }
};

//----------------------------------------------------------------

struct GpuEventAllocator
{
    virtual stdbool eventCreate(const GpuContext& context, bool timingEnabled, GpuEventOwner& result, stdParsNull) =0;
};

//================================================================
//
// GpuEventRecording
//
//================================================================

struct GpuEventRecording
{
    //
    // Captures in the event all the work submitted to the stream so far.
    //
    // This function may be called multiple times for the same event,
    // in this case the new work set replaces the old one,
    // so waiting on such event will wait for the new work set to finish.
    //

    virtual stdbool recordEvent(const GpuEvent& event, const GpuStream& stream, stdParsNull) =0;

    //
    // Puts into the other stream a command to wait for
    // the completion the given event's work set.
    //
    // The wait is performed on GPU without CPU sync.
    //

    virtual stdbool putEventDependency(const GpuEvent& event, const GpuStream& stream, stdParsNull) =0;
};

//================================================================
//
// GpuEventWaiting
//
//================================================================

struct GpuEventWaiting
{
    //
    // Waits on CPU for the event's work set to complete.
    //

    virtual stdbool waitEvent(const GpuEvent& event, bool& realWaitHappened, stdParsNull) =0;

    inline stdbool waitEvent(const GpuEvent& event, stdParsNull)
        {bool tmp = false; return waitEvent(event, tmp, stdPassNullThru);}

    //
    // Compute time elapsed between two events.
    // Both events should be finished, otherwise error is returned.
    //

    virtual stdbool eventElapsedTime(const GpuEvent& event1, const GpuEvent& event2, float32& time, stdParsNull) =0;
};

//================================================================
//
// GpuTransfer
//
// Functions to copy arrays/matrices between CPU <==> GPU memory.
//
// Virtual functions are base interface functions, working with raw byte data:
// sizeX and pitch are specified in bytes;
// memory pointers are specified as unsigned address space integers.
//
// Inline functions are convenience thunks for using in application code:
// the functions take Array/Matrix arguments, so sizeX and pitch
// are specified in elements, and normal typed pointers are used.
//
//================================================================

struct GpuTransfer
{

    //
    // Array
    //

    #define TMP_COPY_ARRAY_PROTO(funcName, SrcAddr, DstAddr) \
        \
        virtual stdbool funcName \
        ( \
            SrcAddr srcAddr, \
            DstAddr dstAddr, \
            Space size, \
            const GpuStream& stream, \
            stdParsNull \
        ) \
        =0; \

    TMP_COPY_ARRAY_PROTO(copyArrayCpuCpu, CpuAddrU, CpuAddrU)
    TMP_COPY_ARRAY_PROTO(copyArrayCpuGpu, CpuAddrU, GpuAddrU)
    TMP_COPY_ARRAY_PROTO(copyArrayGpuCpu, GpuAddrU, CpuAddrU)
    TMP_COPY_ARRAY_PROTO(copyArrayGpuGpu, GpuAddrU, GpuAddrU)

    #undef TMP_COPY_ARRAY_PROTO

    //
    // Matrix
    //

    #define TMP_COPY_MATRIX_PROTO(funcName, SrcAddr, DstAddr) \
        \
        virtual stdbool funcName \
        ( \
            SrcAddr srcAddr, Space srcBytePitch, \
            DstAddr dstAddr, Space dstBytePitch, \
            Space byteSizeX, Space sizeY, \
            const GpuStream& stream, \
            stdParsNull \
        ) \
        =0; \
        \

    TMP_COPY_MATRIX_PROTO(copyMatrixCpuCpu, CpuAddrU, CpuAddrU)
    TMP_COPY_MATRIX_PROTO(copyMatrixCpuGpu, CpuAddrU, GpuAddrU)
    TMP_COPY_MATRIX_PROTO(copyMatrixGpuCpu, GpuAddrU, CpuAddrU)
    TMP_COPY_MATRIX_PROTO(copyMatrixGpuGpu, GpuAddrU, GpuAddrU)

    #undef TMP_COPY_MATRIX_PROTO

};

//================================================================
//
// enqueueCopy<Array>
//
//================================================================

#define TMP_COPY_ARRAY_INLINE(funcName, SrcAddr, DstAddr, SrcPtr, DstPtr, pureGpuValue) \
    \
    template <typename Src, typename Dst, typename Kit> \
    inline stdbool enqueueCopy(const ArrayEx<SrcPtr(Src)>& src, const ArrayEx<DstPtr(Dst)>& dst, const GpuStream& stream, bool& pureGpu, stdPars(Kit)) \
    { \
        COMPILE_ASSERT(TYPE_EQUAL(Src, Dst) || TYPE_EQUAL(Src, const Dst)); \
        \
        REQUIRE(equalSize(src, dst)); \
        \
        require \
        ( \
            kit.gpuTransfer.funcName \
            ( \
                SrcAddr(src.ptrUnsafeForInternalUseOnly()), \
                DstAddr(dst.ptrUnsafeForInternalUseOnly()), \
                dst.size() * sizeof(Dst), \
                stream, \
                stdPass \
            ) \
        ); \
        \
        pureGpu = pureGpuValue; \
        returnTrue; \
    }

TMP_COPY_ARRAY_INLINE(copyArrayCpuCpu, CpuAddrU, CpuAddrU, CpuPtr, CpuPtr, false)
#if HEXLIB_PLATFORM != 0
TMP_COPY_ARRAY_INLINE(copyArrayCpuGpu, CpuAddrU, GpuAddrU, CpuPtr, GpuPtr, false)
TMP_COPY_ARRAY_INLINE(copyArrayGpuCpu, GpuAddrU, CpuAddrU, GpuPtr, CpuPtr, false)
TMP_COPY_ARRAY_INLINE(copyArrayGpuGpu, GpuAddrU, GpuAddrU, GpuPtr, GpuPtr, true)
#endif

#undef TMP_COPY_ARRAY_INLINE

//================================================================
//
// enqueueCopy<Matrix>
//
//================================================================

#define TMP_COPY_MATRIX_INLINE(funcName, SrcAddr, DstAddr, SrcPtr, DstPtr, pureGpuValue) \
    \
    template <typename Src, typename SrcPitch, typename Dst, typename DstPitch, typename Kit> \
    inline stdbool enqueueCopy(const MatrixEx<SrcPtr(Src), SrcPitch>& src, const MatrixEx<DstPtr(Dst), DstPitch>& dst, const GpuStream& stream, bool& pureGpu, stdPars(Kit)) \
    { \
        COMPILE_ASSERT(TYPE_EQUAL(Src, Dst) || TYPE_EQUAL(Src, const Dst)); \
        \
        REQUIRE(equalSize(src, dst)); \
        \
        require \
        ( \
            kit.gpuTransfer.funcName \
            ( \
                SrcAddr(src.memPtrUnsafeInternalUseOnly()), \
                src.memPitch() * sizeof(Dst), \
                DstAddr(dst.memPtrUnsafeInternalUseOnly()), \
                dst.memPitch() * sizeof(Dst), \
                dst.sizeX() * sizeof(Dst), \
                dst.sizeY(), \
                stream, \
                stdPass \
            ) \
        ); \
        \
        pureGpu = pureGpuValue; \
        returnTrue; \
    }

TMP_COPY_MATRIX_INLINE(copyMatrixCpuCpu, CpuAddrU, CpuAddrU, CpuPtr, CpuPtr, false)
#if HEXLIB_PLATFORM != 0
TMP_COPY_MATRIX_INLINE(copyMatrixCpuGpu, CpuAddrU, GpuAddrU, CpuPtr, GpuPtr, false)
TMP_COPY_MATRIX_INLINE(copyMatrixGpuCpu, GpuAddrU, CpuAddrU, GpuPtr, CpuPtr, false)
TMP_COPY_MATRIX_INLINE(copyMatrixGpuGpu, GpuAddrU, GpuAddrU, GpuPtr, GpuPtr, true)
#endif

#undef TMP_COPY_MATRIX_INLINE

//================================================================
//
// GpuSyncGuardThunk
//
// The convenience sync class:
// Automatically performs the command queue waiting in the destructor.
//
// Used to ensure synchronization even when breaking out of normal control flow.
//
//================================================================

class GpuSyncGuardThunk
{

public:

    inline GpuSyncGuardThunk()
    {
        theStream = 0;
        theSyncStream = 0;
    }

    inline ~GpuSyncGuardThunk()
    {
        waitClear();
    }

    inline void reassign(const GpuStream& stream, GpuStreamWaiting& waitStream)
    {
        if_not (theStream == &stream && theSyncStream == &waitStream)
        {
            waitClear();

            this->theStream = &stream;
            this->theSyncStream = &waitStream;
        }
    }

    inline void waitClear()
    {
        if (theStream != 0)
        {
            stdTraceRoot;
            errorBlock(theSyncStream->waitStream(*theStream, stdPassNullNc));

            theStream = 0;
            theSyncStream = 0;
        }
    }

    inline void cancelSync()
    {
        theStream = 0;
        theSyncStream = 0;
    }

private:

    const GpuStream* theStream;
    GpuStreamWaiting* theSyncStream;

};

//================================================================
//
// GpuCopyThunk
//
// Safe transfer interface: Waits for command queue completion in destructor.
//
//================================================================

class GpuCopyThunk
{

public:

    template <typename Src, typename Dst, typename Kit>
    inline stdbool operator()(const Src& src, const Dst& dst, const GpuStream& stream, stdPars(Kit))
    {
        if (kit.dataProcessing)
        {
            bool pureGpu = false;

            require(enqueueCopy(src, dst, stream, pureGpu, stdPassThru));

            if (!pureGpu)
                ioGuard.reassign(stream, kit.gpuStreamWaiting);
        }

        returnTrue;
    }

    ////

    template <typename Src, typename Dst, typename Kit>
    inline stdbool operator()(const Src& src, const Dst& dst, stdPars(Kit))
    {
        return operator()(src, dst, kit.gpuCurrentStream, stdPassThru);
    }

    ////

    inline void waitClear()
        {ioGuard.waitClear();}

    inline void cancelSync()
        {ioGuard.cancelSync();}

private:

    GpuSyncGuardThunk ioGuard;

};

//================================================================
//
// GroupCount
//
//================================================================

class GroupCount : public Point3D<Space>
{

public:

    sysinline GroupCount(const Point3D<Space>& p)
        : Point3D<Space>(p) {}

public:

    sysinline GroupCount(Space X)
        : Point3D<Space>(point3D(X, 1, 1)) {}

    sysinline GroupCount(Space X, Space Y)
        : Point3D<Space>(point3D(X, Y, 1)) {}

    sysinline GroupCount(Space X, Space Y, Space Z)
        : Point3D<Space>(point3D(X, Y, Z)) {}

public:

    sysinline GroupCount(const Point<Space>& p)
        : Point3D<Space>(point3D(p.X, p.Y, 1)) {}

};

//================================================================
//
// GpuKernelCalling
//
//================================================================

struct GpuKernelCalling
{
    virtual stdbool callKernel
    (
        const Point3D<Space>& groupCount,
        const Point<Space>& threadCount,
        uint32 dbgElemCount,
        const struct GpuKernelLink& kernelLink,
        const void* paramPtr, size_t paramSize,
        const GpuStream& stream,
        stdParsNull
    )
    =0;

    template <typename Params>
    inline stdbool callKernel
    (
        const GroupCount& groupCount,
        const Point<Space>& threadCount,
        uint32 dbgElemCount,
        const GpuKernelLink& kernelLink,
        const Params& params,
        const GpuStream& stream,
        stdParsNull
    )
    {
        return callKernel(groupCount, threadCount, dbgElemCount, kernelLink, &params, sizeof(params), stream, stdPassNullThru);
    }

};
