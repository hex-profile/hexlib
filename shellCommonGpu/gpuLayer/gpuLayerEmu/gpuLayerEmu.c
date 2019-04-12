#if HEXLIB_PLATFORM == 0

#include "gpuLayerEmu.h"
#include <omp.h>
#include <memory>
#include <string.h>

#include "errorLog/errorLog.h"
#include "gpuLayer/gpuLayerEmu/uniformPartition.h"
#include "gpuLayer/gpuLayerEmu/emuSampler.h"
#include "dataAlloc/matrixMemory.h"
#include "errorLog/debugBreak.h"
#include "storage/rememberCleanup.h"
#include "interfaces/syncObjects.h"
#include "interfaces/threadManager.h"
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

bool EmuInitApiThunk::initialize(stdNullPars)
{
    stdBegin;
    stdEnd;
}

//================================================================
//
// EmuInitApiThunk::getDeviceCount
//
//================================================================

bool EmuInitApiThunk::getDeviceCount(int32& deviceCount, stdNullPars)
{
    stdBegin;
    deviceCount = 1;
    stdEnd;
}

//================================================================
//
// EmuInitApiThunk::getProperties
//
//================================================================

bool EmuInitApiThunk::getProperties(int32 deviceIndex, GpuProperties& properties, stdNullPars)
{
    stdBegin;

    using namespace emuMultiProc;

    REQUIRE(deviceIndex == 0);

    properties.gpuHardware = false;
    properties.multiprocessorCount = 1;
    properties.clockRate = 2e9f;
    properties.totalThroughput = 2e9f;
    properties.samplerBaseAlignment = EMU_TEX_BASE_ALIGNMENT;
    properties.samplerRowAlignment = EMU_TEX_ROW_ALIGNMENT;
    properties.maxGroupCount = point3D(SpaceU(typeMax<Space>()));
    properties.maxThreadCount = point(EMU_MAX_THREAD_COUNT_X, EMU_MAX_THREAD_COUNT_Y);
    properties.maxGroupArea = EMU_MAX_THREAD_COUNT;

    stdEnd;
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

    KIT_COMBINE3(CreateKit, ErrorLogKit, MallocKit, ThreadManagerKit);

    //----------------------------------------------------------------
    //
    // Creation stage
    //
    // If creation fails, the other methods should NOT called (except destructor).
    // The class doesn't check it.
    //
    //----------------------------------------------------------------

public:

    bool create(const GpuProperties& gpuProperties, stdPars(CreateKit))
    {
        stdBegin;

        Space cpuCount = emuMultiProc::getCpuCount();


        require(emulator.create(cpuCount, stdPassThru));
        REMEMBER_CLEANUP1_EX(emulatorCleanup, emulator.destroy(), emuMultiProc::EmuMultiProc&, emulator);

        require(kit.threadManager.createCriticalSection(emuLock, stdPass));
        REMEMBER_CLEANUP1_EX(emuLockCleanup, emuLock.clear(), CriticalSection&, emuLock);

        this->gpuProperties = gpuProperties;
        emulatorThreadCount = cpuCount;
        emulatorCleanup.cancel();

        emuLockCleanup.cancel();

        stdEnd;
    }

    //----------------------------------------------------------------
    //
    // Execution stage
    //
    //----------------------------------------------------------------

public:

    bool emuLaunchKernel
    (
        const Point3D<Space>& groupCount,
        const Point<Space>& threadCount,
        EmuKernelFunc* kernel,
        const void* userParams,
        stdPars(ErrorLogKit)
    )
    {
        stdBegin;

        CRITSEC_GUARD(emuLock);
        return emulator.launchKernel(groupCount, threadCount, kernel, userParams, stdPassThru);

        stdEnd;
    }

public:

    inline Space emuThreadCount() const
    {
        return emulatorThreadCount;
    }

private:

    GpuProperties gpuProperties;
    CriticalSection emuLock;

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
    COMPILE_ASSERT(sizeof(ContextEx*) <= sizeof(GpuContext));
    return ** (ContextEx**) &context;
}

//================================================================
//
// EmuInitApiThunk::createContext
//
//================================================================

bool EmuInitApiThunk::createContext(int32 deviceIndex, GpuContextOwner& result, void*& baseContext, stdNullPars)
{
    stdBegin;

    result.clear();
    baseContext = 0;

    ////

    ContextEx* ctx = new (std::nothrow) ContextEx;
    REQUIRE(ctx != 0);
    REMEMBER_CLEANUP1_EX(contextAllocCleanup, delete ctx, ContextEx*, ctx);

    //
    // Standard malloc allocator
    //

    GpuProperties gpuProperties;
    require(getProperties(deviceIndex, gpuProperties, stdPass));
    require(ctx->create(gpuProperties, stdPass));

    ////

    GpuContextDeallocContext& deallocContext = result.owner.replace(destroyContext);

    COMPILE_ASSERT(sizeof(ContextEx*) <= sizeof(deallocContext));
    * (ContextEx**) &deallocContext = ctx;

    GpuContext& resultBase = result;
    COMPILE_ASSERT(sizeof(ContextEx*) <= sizeof(resultBase));
    * (ContextEx**) &resultBase = ctx;

    ////

    contextAllocCleanup.cancel();

    ////

    stdEnd;
}

//================================================================
//
// EmuInitApiThunk::destroyContext
//
//================================================================

void EmuInitApiThunk::destroyContext(GpuContextDeallocContext& deallocContext)
{
    COMPILE_ASSERT(sizeof(ContextEx*) <= sizeof(deallocContext));
    ContextEx*& context = * (ContextEx**) &deallocContext;
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

bool emuAllocateTexture(Space sizeX, Space sizeY, GpuChannelType chanType, int rank, Byte*& sysAllocPtr, EmuTexture& result, stdPars(ErrorLogKit))
{
    stdBegin;

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

    stdEnd;
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

bool EmuInitApiThunk::createTexture(const GpuContext& context, const Point<Space>& size, GpuChannelType chanType, int rank, GpuTextureOwner& result, stdNullPars)
{
    stdBegin;

    ++textureAllocCount;

    ////

    result.clear();

    GpuTexture& gpuTexture = result;

    COMPILE_ASSERT(sizeof(EmuTexture) <= sizeof(GpuTexture));
    EmuTexture& emuTexture = (EmuTexture&) gpuTexture;

    ////

    Byte* sysAllocPtr = 0;
    require(emuAllocateTexture(size.X, size.Y, chanType, rank, sysAllocPtr, emuTexture, stdPass));

    ////

    GpuTextureDeallocContext& deallocContext = result.owner.replace(destroyTexture);
    COMPILE_ASSERT(sizeof(EmuTextureContext) <= sizeof(deallocContext));
    EmuTextureContext& emuTextureContext = (EmuTextureContext&) deallocContext;
    emuTextureContext.allocPtr = sysAllocPtr;

    ////

    stdEnd;
}

//================================================================
//
// EmuInitApiThunk::destroyTexture
//
//================================================================

void EmuInitApiThunk::destroyTexture(GpuTextureDeallocContext& deallocContext)
{
    COMPILE_ASSERT(sizeof(EmuTextureContext) <= sizeof(deallocContext));
    EmuTextureContext& context = (EmuTextureContext&) deallocContext;

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

    bool create(const GpuContext& context, stdPars(ErrorLogExKit))
        {this->context = context; return true;}

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
    COMPILE_ASSERT(sizeof(StreamEx*) <= sizeof(GpuStream));
    return ** (StreamEx**) &stream;
}

//================================================================
//
// EmuInitApiThunk::createStream
//
//================================================================

bool EmuInitApiThunk::createStream(const GpuContext& context, bool nullStream, GpuStreamOwner& result, void*& baseStream, stdNullPars)
{
    stdBegin;

    result.clear();
    baseStream = 0;

    ////

    StreamEx* streamEx = new (std::nothrow) StreamEx;
    REQUIRE(streamEx != 0);
    REMEMBER_CLEANUP1_EX(streamAllocCleanup, delete streamEx, StreamEx*, streamEx);

    require(streamEx->create(context, stdPass));

    ////

    GpuStreamDeallocContext& deallocContext = result.owner.replace(destroyStream);

    COMPILE_ASSERT(sizeof(StreamEx*) <= sizeof(deallocContext));
    * (StreamEx**) &deallocContext = streamEx;

    GpuStream& resultBase = result;
    COMPILE_ASSERT(sizeof(StreamEx*) <= sizeof(resultBase));
    * (StreamEx**) &resultBase = streamEx;

    ////

    streamAllocCleanup.cancel();

    stdEnd;
}

//================================================================
//
// EmuInitApiThunk::destroyStream
//
//================================================================

void EmuInitApiThunk::destroyStream(GpuStreamDeallocContext& deallocContext)
{
    COMPILE_ASSERT(sizeof(StreamEx*) <= sizeof(deallocContext));
    StreamEx*& stream = * (StreamEx**) &deallocContext;
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

KIT_CREATE3(
    CopyArrayParams,
    ArrayPtr(Byte), srcPtr,
    ArrayPtr(Byte), dstPtr,
    Space, byteSize
);

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

    ArrayPtr(Byte) src = o.srcPtr + byteStart;
    ArrayPtr(Byte) dst = o.dstPtr + byteStart;

    ////

    if (byteCount)
    {
        // performance check
        devDebugCheck(isPtrAligned<granularity>(src));
        devDebugCheck(isPtrAligned<granularity>(dst));
    }

    ////

    memcpy(unsafePtr(dst, byteCount), unsafePtr(src, byteCount), byteCount);
}

//================================================================
//
// genericArrayCopy
//
//================================================================

inline bool genericArrayCopy
(
    CpuAddrU srcAddr, CpuAddrU dstAddr, Space byteSize,
    ContextEx& ctx,
    stdPars(ErrorLogKit)
)
{
    stdBegin;

    REQUIRE(byteSize >= 0);

    ////

    ArrayPtr(Byte) srcPtr = ArrayPtrCreate(Byte, (Byte*) srcAddr, byteSize, DbgptrArrayPreconditions());
    ArrayPtr(Byte) dstPtr = ArrayPtrCreate(Byte, (Byte*) dstAddr, byteSize, DbgptrArrayPreconditions());

    CopyArrayParams params(srcPtr, dstPtr, byteSize);

    ////

    require
    (
        ctx.emuLaunchKernel
        (
            point3D(1, ctx.emuThreadCount(), 1),
            point(1, 1),
            (EmuKernelFunc*) copyArrayKernelCode,
            &params,
            stdPassThru
        )
    );

    stdEnd;
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
    bool EmuExecApiThunk::funcName(SrcAddr srcAddr, DstAddr dstAddr, Space size, const GpuStream& stream, stdNullPars) \
    { \
        if (gpuEnqueueMode == GpuEnqueueNormal) \
            require(genericArrayCopy(srcAddr, dstAddr, size, uncast(uncast(stream).getContext()), stdPassThru)); \
        \
        return true; \
    }

TMP_MACRO(copyArrayCpuCpu, CpuAddrU, CpuAddrU);
TMP_MACRO(copyArrayCpuGpu, CpuAddrU, GpuAddrU);
TMP_MACRO(copyArrayGpuCpu, GpuAddrU, CpuAddrU);
TMP_MACRO(copyArrayGpuGpu, GpuAddrU, GpuAddrU);

#undef TMP_MACRO

//================================================================
//
// CopyMatrixParams
//
//================================================================

KIT_CREATE6(
    CopyMatrixParams,
    CpuAddrU, srcPtr, Space, srcBytePitch,
    CpuAddrU, dstPtr, Space, dstBytePitch,
    Space, byteSizeX, Space, sizeY
);

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

    for (Space k = 0; k < rowCount; ++k)
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

inline bool genericMatrixCopy
(
    CpuAddrU srcPtr, Space srcBytePitch,
    CpuAddrU dstPtr, Space dstBytePitch,
    Space byteSizeX, Space sizeY,
    ContextEx& ctx,
    stdPars(ErrorLogKit)
)
{
    stdBegin;

    CopyMatrixParams params(srcPtr, srcBytePitch, dstPtr, dstBytePitch, byteSizeX, sizeY);

    require
    (
        ctx.emuLaunchKernel
        (
            point3D(1, ctx.emuThreadCount(), 1),
            point(1, 1),
            (EmuKernelFunc*) copyMatrixKernelCode,
            &params,
            stdPassThru
        )
    );

    stdEnd;
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
    bool EmuExecApiThunk::funcName \
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
        return true; \
    }

TMP_MACRO(copyMatrixCpuCpu, CpuAddrU, CpuAddrU);
TMP_MACRO(copyMatrixCpuGpu, CpuAddrU, GpuAddrU);
TMP_MACRO(copyMatrixGpuCpu, GpuAddrU, CpuAddrU);
TMP_MACRO(copyMatrixGpuGpu, GpuAddrU, GpuAddrU);

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

bool EmuExecApiThunk::setSamplerArray
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
    return emuSetSamplerArray(sampler, arrayAddr, arrayByteSize,
        chanType, rank, borderMode, linearInterpolation, readNormalizedFloat, normalizedCoords, stdPassThru);
}

//================================================================
//
// EmuExecApiThunk::setSamplerImage
//
//================================================================

bool EmuExecApiThunk::setSamplerImage
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

bool EmuExecApiThunk::callKernel
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
    stdBegin;

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

    stdEnd;
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

bool EmuExecApiThunk::waitStream(const GpuStream& stream, stdNullPars)
    {return true;}

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Events
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

bool EmuExecApiThunk::putEvent(const GpuEvent& event, const GpuStream& stream, stdNullPars)
{
    return true;
}

bool EmuExecApiThunk::putEventDependency(const GpuEvent& event, const GpuStream& stream, stdNullPars)
{
    return true;
}

bool EmuExecApiThunk::checkEvent(const GpuEvent& event, stdNullPars)
{
    return true;
}

bool EmuExecApiThunk::waitEvent(const GpuEvent& event, bool& realWaitHappened, stdNullPars)
{
    realWaitHappened = false;
    return true;
}

bool EmuExecApiThunk::eventElapsedTime(const GpuEvent& event1, const GpuEvent& event2, float32& time, stdNullPars)
{
    time = 0;
    return true;
}

//----------------------------------------------------------------

#endif
