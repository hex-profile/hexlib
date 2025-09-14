#if HEXLIB_PLATFORM == 0

#include "gpuLayerEmu.h"
#include <omp.h>
#include <memory>
#include <string.h>

#include "errorLog/errorLog.h"
#include "gpuSupport/uniformPartition.h"
#include "gpuLayer/gpuLayerEmu/emuSampler.h"
#include "dataAlloc/matrixMemory.h"
#include "errorLog/debugBreak.h"
#include "storage/rememberCleanup.h"
#include "interfaces/syncObjects.h"
#include "threads/threads.h"
#include "data/spacex.h"

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Init
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

static const Space EMU_TEX_BASE_ALIGNMENT = 512;
static const Space EMU_TEX_ROW_ALIGNMENT = 32;

//================================================================
//
// EmuInitApiThunk::initialize
//
//================================================================

stdbool EmuInitApiThunk::initialize(stdNullPars)
{
    returnTrue;
}

//================================================================
//
// EmuInitApiThunk::getDeviceCount
//
//================================================================

stdbool EmuInitApiThunk::getDeviceCount(int32& deviceCount, stdNullPars)
{
    deviceCount = 1;
    returnTrue;
}

//================================================================
//
// EmuInitApiThunk::getProperties
//
//================================================================

stdbool EmuInitApiThunk::getProperties(int32 deviceIndex, GpuProperties& properties, stdNullPars)
{
    using namespace emuMultiProc;

    REQUIRE(deviceIndex == 0);

    properties.occupancyThreadsGood = 2048 * 32;
    properties.occupancyThreadsMax = 2048 * 64;
    properties.totalThroughput = 2e9f;
    properties.samplerAndFastTransferBaseAlignment = EMU_TEX_BASE_ALIGNMENT;
    properties.samplerRowAlignment = EMU_TEX_ROW_ALIGNMENT;
    properties.maxGroupCount = point3D(SpaceU(typeMax<Space>()));
    properties.maxThreadCount = point(EMU_MAX_THREAD_COUNT_X, EMU_MAX_THREAD_COUNT_Y);
    properties.maxGroupArea = EMU_MAX_THREAD_COUNT;

    returnTrue;
}

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Context
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// ContextEx
//
//================================================================

class ContextEx
{

private:

    using CreateKit = KitCombine<ErrorLogKit, MallocKit>;

    //----------------------------------------------------------------
    //
    // Creation stage
    //
    // If creation fails, the other methods should NOT be called (except the destructor).
    // The class doesn't check it.
    //
    //----------------------------------------------------------------

public:

    stdbool create(const GpuProperties& gpuProperties, stdPars(CreateKit))
    {
        Space cpuCount = emuMultiProc::getCpuCount();

        require(emulator.create(cpuCount, stdPassThru));
        REMEMBER_CLEANUP_EX(emulatorCleanup, emulator.destroy());

        require(mutexCreate(emuLock, stdPass));
        REMEMBER_CLEANUP_EX(emuLockCleanup, emuLock.clear());

        this->gpuProperties = gpuProperties;
        emulatorThreadCount = cpuCount;
        emulatorCleanup.cancel();

        emuLockCleanup.cancel();

        returnTrue;
    }

    //----------------------------------------------------------------
    //
    // Execution stage
    //
    //----------------------------------------------------------------

public:

    stdbool emuLaunchKernel
    (
        const Point3D<Space>& groupCount,
        const Point<Space>& threadCount,
        EmuKernelFunc* kernel,
        const void* userParams,
        stdPars(ErrorLogKit)
    )
    {
        MUTEX_GUARD(emuLock);
        return emulator.launchKernel(groupCount, threadCount, kernel, userParams, stdPassThru);
    }

public:

    inline Space emuThreadCount() const
    {
        return emulatorThreadCount;
    }

private:

    GpuProperties gpuProperties;
    Mutex emuLock;

    emuMultiProc::EmuMultiProc emulator;
    Space emulatorThreadCount = 0;

};

//================================================================
//
// uncast
//
//================================================================

inline ContextEx& uncast(const GpuContext& context)
{
    return *context.recast<ContextEx*>();
}

//================================================================
//
// EmuInitApiThunk::createContext
//
//================================================================

stdbool EmuInitApiThunk::createContext(int32 deviceIndex, GpuScheduling gpuScheduling, GpuContextOwner& result, void*& baseContext, stdNullPars)
{
    result.clear();
    baseContext = 0;

    ////

    ContextEx* ctx = new (std::nothrow) ContextEx;
    REQUIRE(ctx != 0);
    REMEMBER_CLEANUP_EX(contextAllocCleanup, delete ctx);

    //
    // Standard malloc allocator
    //

    GpuProperties gpuProperties;
    require(getProperties(deviceIndex, gpuProperties, stdPass));
    require(ctx->create(gpuProperties, stdPass));

    ////

    GpuContextDeallocContext& deallocContext = result.owner.replace(destroyContext);

    deallocContext.recast<ContextEx*>() = ctx;

    GpuContext& resultBase = result;
    resultBase.recast<ContextEx*>() = ctx;

    ////

    contextAllocCleanup.cancel();

    ////

    returnTrue;
}

//================================================================
//
// EmuInitApiThunk::destroyContext
//
//================================================================

void EmuInitApiThunk::destroyContext(GpuContextDeallocContext& deallocContext)
{
    auto& context = deallocContext.recast<ContextEx*>();
    delete context;
    context = 0;
}

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Texture allocation
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// EmuTexture
//
//================================================================

struct EmuTexture
{
    CpuAddrU imageBaseAddr;
    Space imageBytePitch;
    Point<Space> imageSize;

    GpuChannelType chanType;
    int rank;

    inline void assignNull()
    {
        imageBaseAddr = 0;
        imageBytePitch = 0;
        imageSize = point(0);

        chanType = GpuChannelInt8;
        rank = 0;
    }
};

//================================================================
//
// emuAllocateTexture
//
//================================================================

stdbool emuAllocateTexture(Space sizeX, Space sizeY, GpuChannelType chanType, int rank, Byte*& sysAllocPtr, EmuTexture& result, stdPars(ErrorLogKit))
{
    REQUIRE(sizeX >= 0 && sizeY >= 0);

    //
    // element size
    //

    Space elemSize = 0;

    #define TMP_MACRO(cType, type, o) \
        if (cType == chanType) elemSize = rank * sizeof(type);

    GPU_CHANTYPE_FOREACH(TMP_MACRO, o)

    #undef TMP_MACRO

    REQUIRE(elemSize != 0);

    //
    // alignments
    //

    const Space baseByteAlignment = EMU_TEX_BASE_ALIGNMENT;
    const Space rowByteAlignment = EMU_TEX_ROW_ALIGNMENT;

    COMPILE_ASSERT(COMPILE_IS_POWER2(baseByteAlignment) && COMPILE_IS_POWER2(rowByteAlignment));
    COMPILE_ASSERT(1 <= rowByteAlignment && rowByteAlignment <= baseByteAlignment);

    //
    // row byte size
    //

    Space rowByteSize = 0;
    REQUIRE(safeMul(sizeX, elemSize, rowByteSize));

    //
    // aligned row byte size
    //

    REQUIRE(rowByteAlignment % elemSize == 0); // divides evenly

    const Space rowByteAlignmentMask = rowByteAlignment - 1;

    Space rowByteSizePlusMask = 0;
    REQUIRE(safeAdd(rowByteSize, rowByteAlignmentMask, rowByteSizePlusMask));

    Space alignedRowByteSize = rowByteSizePlusMask & (~rowByteAlignmentMask);
    REQUIRE(alignedRowByteSize >= rowByteSize); // self-check

    //
    // allocation byte size
    //

    Space byteAllocSize = 0;
    REQUIRE(safeMul(alignedRowByteSize, sizeY, byteAllocSize));

    //
    // allocate with adjustment for base address alignment
    // save allocation pointer
    //

    Space baseAlignMask = baseByteAlignment - 1;
    Space maxCorrection = baseAlignMask;

    Space sysAllocSize = 0;
    REQUIRE(safeAdd(byteAllocSize, maxCorrection, sysAllocSize));

    sysAllocPtr = (Byte*) malloc(sysAllocSize);
    REQUIRE(sysAllocPtr != 0);

    CpuAddrU allocPtr = (CpuAddrU) sysAllocPtr;
    CpuAddrU alignedPtr = (allocPtr + baseAlignMask) & ~baseAlignMask; // overflow impossible

    //
    //
    //

    result.imageBaseAddr = alignedPtr;
    result.imageBytePitch = alignedRowByteSize;
    result.imageSize = point(sizeX, sizeY);

    returnTrue;
}

//================================================================
//
// EmuTextureContext
//
//================================================================

struct EmuTextureContext
{
    Byte* allocPtr;
};

//================================================================
//
// EmuInitApiThunk::createTexture
//
//================================================================

stdbool EmuInitApiThunk::createTexture(const GpuContext& context, const Point<Space>& size, GpuChannelType chanType, int rank, GpuTextureOwner& result, stdNullPars)
{
    ++textureAllocCount;

    ////

    result.clear();

    GpuTexture& gpuTexture = result;

    auto& emuTexture = gpuTexture.recast<EmuTexture>();

    ////

    Byte* sysAllocPtr = 0;
    require(emuAllocateTexture(size.X, size.Y, chanType, rank, sysAllocPtr, emuTexture, stdPass));

    ////

    GpuTextureDeallocContext& deallocContext = result.owner.replace(destroyTexture);
    auto& emuTextureContext = deallocContext.recast<EmuTextureContext>();
    emuTextureContext.allocPtr = sysAllocPtr;

    ////

    returnTrue;
}

//================================================================
//
// EmuInitApiThunk::destroyTexture
//
//================================================================

void EmuInitApiThunk::destroyTexture(GpuTextureDeallocContext& deallocContext)
{
    auto& context = deallocContext.recast<EmuTextureContext>();

    if (context.allocPtr != 0)
    {
        free(context.allocPtr);
        context.allocPtr = 0;
    }
}

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Stream allocation
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// StreamEx
//
//================================================================

struct StreamEx
{

public:

    ~StreamEx() {destroy();}

    stdbool create(const GpuContext& context, stdPars(MsgLogExKit))
        {this->context = context; returnTrue;}

    void destroy()
        {}

public:

    sysinline const GpuContext& getContext() const
        {return context;}

public:

    GpuContext context;

};

//================================================================
//
// uncast
//
//================================================================

inline StreamEx& uncast(const GpuStream& stream)
{
    return *stream.recast<StreamEx*>();
}

//================================================================
//
// getNativeHandle
//
//================================================================

void* getNativeHandle(const GpuStream& stream)
{
    return nullptr;
}

//================================================================
//
// EmuInitApiThunk::createStream
//
//================================================================

stdbool EmuInitApiThunk::createStream(const GpuContext& context, bool nullStream, GpuStreamOwner& result, stdNullPars)
{
    result.clear();

    ////

    StreamEx* streamEx = new (std::nothrow) StreamEx;
    REQUIRE(streamEx != 0);
    REMEMBER_CLEANUP_EX(streamAllocCleanup, delete streamEx);

    require(streamEx->create(context, stdPass));

    ////

    GpuStreamDeallocContext& deallocContext = result.owner.replace(destroyStream);

    deallocContext.recast<StreamEx*>() = streamEx;

    GpuStream& resultBase = result;
    resultBase.recast<StreamEx*>() = streamEx;

    ////

    streamAllocCleanup.cancel();

    returnTrue;
}

//================================================================
//
// EmuInitApiThunk::destroyStream
//
//================================================================

void EmuInitApiThunk::destroyStream(GpuStreamDeallocContext& deallocContext)
{
    auto& stream = deallocContext.recast<StreamEx*>();
    delete stream;
    stream = 0;
}

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Transfers
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// CopyArrayParams
//
//================================================================

struct CopyArrayParams
{
    ArrayPtr(Byte) srcPtr;
    ArrayPtr(Byte) dstPtr;
    Space byteSize;
};

//================================================================
//
// copyArrayKernel
//
//================================================================

devDefineKernel(copyArrayKernel, CopyArrayParams, o)
{
    devDebugCheck(o.byteSize >= 0);
    devDebugCheck(devGroupCountY >= 1);

    ////

    static const Space granBits = 4;
    static const Space granularity = 1 << granBits; // CPU SIMD: 16 bytes

    ////

    Space byteSizeAdded = 0;
    devDebugCheck(safeAdd(o.byteSize, granularity-1, byteSizeAdded));
    Space granSize = byteSizeAdded >> granBits;

    ////

    UniformPartition<Space> partition(granSize, devGroupCountY);

    Space granStart = partition.nthOrg(devGroupY);
    Space granCount = partition.nthSize(devGroupY);

    ////

    Space byteStart = granStart << granBits;
    Space byteCount = granCount << granBits;

    ////

    byteStart = clampMax(byteStart, o.byteSize);
    byteCount = clampMax(byteCount, o.byteSize - byteStart);

    ////

    auto src = o.srcPtr + byteStart;
    auto dst = o.dstPtr + byteStart;

    ////

    memcpy(unsafePtr(dst, byteCount), unsafePtr(src, byteCount), byteCount);
}

//================================================================
//
// genericArrayCopy
//
//================================================================

inline stdbool genericArrayCopy
(
    CpuAddrU srcAddr, CpuAddrU dstAddr, Space byteSize,
    ContextEx& ctx,
    stdPars(ErrorLogKit)
)
{
    REQUIRE(byteSize >= 0);

    ////

    auto srcPtr = ArrayPtrCreate(Byte, (Byte*) srcAddr, byteSize, DbgptrArrayPreconditions());
    auto dstPtr = ArrayPtrCreate(Byte, (Byte*) dstAddr, byteSize, DbgptrArrayPreconditions());

    CopyArrayParams params{srcPtr, dstPtr, byteSize};

    ////

    require
    (
        ctx.emuLaunchKernel
        (
            point3D(1, ctx.emuThreadCount(), 1),
            point(1, 1),
            (EmuKernelFunc*) (void*) copyArrayKernelCode,
            &params,
            stdPassThru
        )
    );

    returnTrue;
}

//================================================================
//
// EmuExecApiThunk::copyArrayCpuCpu
// EmuExecApiThunk::copyArrayCpuGpu
// EmuExecApiThunk::copyArrayGpuCpu
// EmuExecApiThunk::copyArrayGpuGpu
//
//================================================================

#define TMP_MACRO(funcName, SrcAddr, DstAddr) \
    \
    stdbool EmuExecApiThunk::funcName(SrcAddr srcAddr, DstAddr dstAddr, Space size, const GpuStream& stream, stdNullPars) \
    { \
        if (gpuEnqueueMode == GpuEnqueueNormal) \
            require(genericArrayCopy(srcAddr, dstAddr, size, uncast(uncast(stream).getContext()), stdPassThru)); \
        \
        returnTrue; \
    }

TMP_MACRO(copyArrayCpuCpu, CpuAddrU, CpuAddrU)
TMP_MACRO(copyArrayCpuGpu, CpuAddrU, GpuAddrU)
TMP_MACRO(copyArrayGpuCpu, GpuAddrU, CpuAddrU)
TMP_MACRO(copyArrayGpuGpu, GpuAddrU, GpuAddrU)

#undef TMP_MACRO

//================================================================
//
// CopyMatrixParams
//
//================================================================

struct CopyMatrixParams
{
    CpuAddrU srcPtr;
    Space srcBytePitch;

    CpuAddrU dstPtr;
    Space dstBytePitch;

    Space byteSizeX;
    Space sizeY;
};

//================================================================
//
// copyMatrixKernel
//
//================================================================

devDefineKernel(copyMatrixKernel, CopyMatrixParams, o)
{
    devDebugCheck(o.byteSizeX >= 0);

    devDebugCheck(devGroupCountY >= 1);
    UniformPartition<Space> partition(o.sizeY, devGroupCountY);

    Space rowStart = partition.nthOrg(devGroupY);
    Space rowCount = partition.nthSize(devGroupY);

    for_count (k, rowCount)
    {
        Space Y = rowStart + k;

        CpuAddrU src = o.srcPtr + Space(Y * o.srcBytePitch);
        CpuAddrU dst = o.dstPtr + Space(Y * o.dstBytePitch);

        memcpy((void*) dst, (void*) src, o.byteSizeX);
    }
}

//================================================================
//
// genericMatrixCopy
//
//================================================================

inline stdbool genericMatrixCopy
(
    CpuAddrU srcPtr, Space srcBytePitch,
    CpuAddrU dstPtr, Space dstBytePitch,
    Space byteSizeX, Space sizeY,
    ContextEx& ctx,
    stdPars(ErrorLogKit)
)
{
    CopyMatrixParams params{srcPtr, srcBytePitch, dstPtr, dstBytePitch, byteSizeX, sizeY};

    require
    (
        ctx.emuLaunchKernel
        (
            point3D(1, ctx.emuThreadCount(), 1),
            point(1, 1),
            (EmuKernelFunc*) (void*) copyMatrixKernelCode,
            &params,
            stdPassThru
        )
    );

    returnTrue;
}

//================================================================
//
// EmuExecApiThunk::copyMatrixCpuCpu
// EmuExecApiThunk::copyMatrixCpuGpu
// EmuExecApiThunk::copyMatrixGpuCpu
// EmuExecApiThunk::copyMatrixGpuGpu
//
//================================================================

#define TMP_MACRO(funcName, SrcAddr, DstAddr) \
    \
    stdbool EmuExecApiThunk::funcName \
    ( \
        SrcAddr srcAddr, Space srcBytePitch, \
        DstAddr dstAddr, Space dstBytePitch, \
        Space byteSizeX, Space sizeY, \
        const GpuStream& stream, \
        stdNullPars \
    ) \
    { \
        if (gpuEnqueueMode == GpuEnqueueNormal) \
        { \
            require \
            ( \
                genericMatrixCopy \
                ( \
                    srcAddr, srcBytePitch, \
                    dstAddr, dstBytePitch, \
                    byteSizeX, sizeY, \
                    uncast(uncast(stream).getContext()), \
                    stdPassThru \
                ) \
            ); \
        } \
        \
        returnTrue; \
    }

TMP_MACRO(copyMatrixCpuCpu, CpuAddrU, CpuAddrU)
TMP_MACRO(copyMatrixCpuGpu, CpuAddrU, GpuAddrU)
TMP_MACRO(copyMatrixGpuCpu, GpuAddrU, CpuAddrU)
TMP_MACRO(copyMatrixGpuGpu, GpuAddrU, GpuAddrU)

#undef TMP_MACRO

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Sampler setup
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// EmuExecApiThunk::setSamplerArray
//
//================================================================

stdbool EmuExecApiThunk::setSamplerArray
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
    stdNullPars
)
{
    return emuSetSamplerArray(sampler, arrayAddr, arrayByteSize,
        chanType, rank, borderMode, linearInterpolation, readNormalizedFloat, normalizedCoords, stdPassThru);
}

//================================================================
//
// EmuExecApiThunk::setSamplerImageEx
//
//================================================================

stdbool EmuExecApiThunk::setSamplerImageEx
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
    stdNullPars
)
{
    return emuSetSamplerImage(sampler, imageBaseAddr, imageBytePitch, imageSize, chanType, rank,
        borderMode, linearInterpolation, readNormalizedFloat, normalizedCoords, stdPassThru);
}

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Kernel launching
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// EmuExecApiThunk::callKernel
//
//================================================================

stdbool EmuExecApiThunk::callKernel
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
    ContextEx& ctx = uncast(uncast(stream).getContext());

    EmuKernelFunc* kernelFunc = kernelLink.func;

    ////

    if (gpuEnqueueMode == GpuEnqueueNormal)
    {
        require
        (
            ctx.emuLaunchKernel
            (
                groupCount,
                threadCount,
                kernelFunc,
                paramPtr,
                stdPassThru
            )
        );
    }

    ////

    returnTrue;
}

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Stream sync
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

stdbool EmuExecApiThunk::waitStream(const GpuStream& stream, stdNullPars)
{
    returnTrue;
}

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Events
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

stdbool EmuExecApiThunk::recordEvent(const GpuEvent& event, const GpuStream& stream, stdNullPars)
{
    returnTrue;
}

stdbool EmuExecApiThunk::putEventDependency(const GpuEvent& event, const GpuStream& stream, stdNullPars)
{
    returnTrue;
}

stdbool EmuExecApiThunk::waitEvent(const GpuEvent& event, bool& realWaitHappened, stdNullPars)
{
    realWaitHappened = false;
    returnTrue;
}

stdbool EmuExecApiThunk::eventElapsedTime(const GpuEvent& event1, const GpuEvent& event2, float32& time, stdNullPars)
{
    time = 0;
    returnTrue;
}

//----------------------------------------------------------------

#endif
