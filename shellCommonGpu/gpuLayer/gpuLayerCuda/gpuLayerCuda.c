#if HEXLIB_PLATFORM == 1

#include "gpuLayerCuda.h"

#include <cuda.h>
#include <stdio.h>
#include <memory>

#include "userOutput/diagnosticKit.h"
#include "allocation/sysAllocAlignShell.h"
#include "errorLog/debugBreak.h"
#include "timer/timer.h"
#include "emptyKernel.h"
#include "userOutput/printMsg.h"
#include "history/historyObj.h"
#include "dataAlloc/arrayObjMem.inl"
#include "numbers/lt/ltBase.h"
#include "numbers/divRound.h"
#include "gpuLayer/gpuLayerCuda/cudaErrorReport.h"

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// CUDA diagnostics
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

constexpr bool initAllocatedBlocks = true;
constexpr bool reportAllocatedBlocks = false;

//================================================================
//
// CUDA_DRVAPI_ERROR_LIST
//
//================================================================

#define CUDA_DRVAPI_ERROR_LIST(action) \
    action(CUDA_SUCCESS) \
    action(CUDA_ERROR_INVALID_VALUE) \
    action(CUDA_ERROR_OUT_OF_MEMORY) \
    action(CUDA_ERROR_NOT_INITIALIZED) \
    action(CUDA_ERROR_DEINITIALIZED) \
    action(CUDA_ERROR_PROFILER_DISABLED) \
    action(CUDA_ERROR_PROFILER_NOT_INITIALIZED) \
    action(CUDA_ERROR_PROFILER_ALREADY_STARTED) \
    action(CUDA_ERROR_PROFILER_ALREADY_STOPPED) \
    action(CUDA_ERROR_NO_DEVICE) \
    action(CUDA_ERROR_INVALID_DEVICE) \
    action(CUDA_ERROR_INVALID_IMAGE) \
    action(CUDA_ERROR_INVALID_CONTEXT) \
    action(CUDA_ERROR_CONTEXT_ALREADY_CURRENT) \
    action(CUDA_ERROR_MAP_FAILED) \
    action(CUDA_ERROR_UNMAP_FAILED) \
    action(CUDA_ERROR_ARRAY_IS_MAPPED) \
    action(CUDA_ERROR_ALREADY_MAPPED) \
    action(CUDA_ERROR_NO_BINARY_FOR_GPU) \
    action(CUDA_ERROR_ALREADY_ACQUIRED) \
    action(CUDA_ERROR_NOT_MAPPED) \
    action(CUDA_ERROR_NOT_MAPPED_AS_ARRAY) \
    action(CUDA_ERROR_NOT_MAPPED_AS_POINTER) \
    action(CUDA_ERROR_ECC_UNCORRECTABLE) \
    action(CUDA_ERROR_UNSUPPORTED_LIMIT) \
    action(CUDA_ERROR_CONTEXT_ALREADY_IN_USE) \
    action(CUDA_ERROR_PEER_ACCESS_UNSUPPORTED) \
    action(CUDA_ERROR_INVALID_PTX) \
    /*action(CUDA_ERROR_INVALID_GRAPHICS_CONTEXT) */ \
    action(CUDA_ERROR_INVALID_SOURCE) \
    action(CUDA_ERROR_FILE_NOT_FOUND) \
    action(CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND) \
    action(CUDA_ERROR_SHARED_OBJECT_INIT_FAILED) \
    action(CUDA_ERROR_OPERATING_SYSTEM) \
    action(CUDA_ERROR_INVALID_HANDLE) \
    action(CUDA_ERROR_NOT_FOUND) \
    action(CUDA_ERROR_NOT_READY) \
    action(CUDA_ERROR_ILLEGAL_ADDRESS) \
    action(CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES) \
    action(CUDA_ERROR_LAUNCH_TIMEOUT) \
    action(CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING) \
    action(CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED) \
    action(CUDA_ERROR_PEER_ACCESS_NOT_ENABLED) \
    action(CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE) \
    action(CUDA_ERROR_CONTEXT_IS_DESTROYED) \
    action(CUDA_ERROR_ASSERT) \
    action(CUDA_ERROR_TOO_MANY_PEERS) \
    action(CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED) \
    action(CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED) \
    action(CUDA_ERROR_HARDWARE_STACK_ERROR) \
    action(CUDA_ERROR_ILLEGAL_INSTRUCTION) \
    action(CUDA_ERROR_MISALIGNED_ADDRESS) \
    action(CUDA_ERROR_INVALID_ADDRESS_SPACE) \
    action(CUDA_ERROR_INVALID_PC) \
    action(CUDA_ERROR_LAUNCH_FAILED) \
    action(CUDA_ERROR_NOT_PERMITTED) \
    action(CUDA_ERROR_NOT_SUPPORTED) \
    action(CUDA_ERROR_UNKNOWN)

//================================================================
//
// formatOutput<CUresult>
//
//================================================================

template <>
void formatOutput(const CUresult& value, FormatOutputStream& outputStream)
{
    const CharType* textDesc = 0;

    ////

    #define TMP_MACRO(err) \
        case err: textDesc = PREP_STRINGIZE(err); break;

    switch (value)
    {
        CUDA_DRVAPI_ERROR_LIST(TMP_MACRO)
    };

    #undef TMP_MACRO

    ////

    if (textDesc)
        formatOutput(textDesc, outputStream);
    else
        formatOutput(int32(value), outputStream);
}

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Utilities
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// ASSIGN_CONVERT
//
//================================================================

#define ASSIGN_CONVERT(dst, src) \
    REQUIRE(convertExact(src, dst));

//================================================================
//
// uncast
//
//================================================================

COMPILE_ASSERT(sizeof(CUevent) <= sizeof(GpuEvent));

inline const CUevent& uncast(const GpuEvent& event)
    {return * (CUevent*) &event;}

inline CUevent& uncast(GpuEvent& event)
    {return * (CUevent*) &event;}

//================================================================
//
// cudaChannelFormat
//
//================================================================

static const CUarray_format cudaChannelXlat[] =
{
    CU_AD_FORMAT_SIGNED_INT8,
    CU_AD_FORMAT_UNSIGNED_INT8,
    CU_AD_FORMAT_SIGNED_INT16,
    CU_AD_FORMAT_UNSIGNED_INT16,
    CU_AD_FORMAT_SIGNED_INT32,
    CU_AD_FORMAT_UNSIGNED_INT32,
    CU_AD_FORMAT_HALF,
    CU_AD_FORMAT_FLOAT,
};

COMPILE_ASSERT(COMPILE_ARRAY_SIZE(cudaChannelXlat) == GpuChannelTypeCount);

//----------------------------------------------------------------

inline bool cudaChannelFormat(GpuChannelType chanType, CUarray_format& result)
{
    require(size_t(chanType) < size_t(COMPILE_ARRAY_SIZE(cudaChannelXlat)));
    result = cudaChannelXlat[size_t(chanType)];
    return true;
}

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Initialization
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// CudaInitApiThunk::initialize
//
//================================================================

bool CudaInitApiThunk::initialize(stdNullPars)
{
    stdBegin;
    REQUIRE_CUDA(cuInit(0));
    stdEnd;
}

//================================================================
//
// CudaInitApiThunk::getDeviceCount
//
//================================================================

bool CudaInitApiThunk::getDeviceCount(int32& deviceCount, stdNullPars)
{
    stdBegin;

    int count = 0;
    REQUIRE_CUDA(cuDeviceGetCount(&count));

    deviceCount = count;

    stdEnd;
}

//================================================================
//
// getCudaCoresPerMultiprocessor
//
// Returns 0 in case of error.
//
//================================================================

int getCudaCoresPerMultiprocessor(int version)
{  
    int cores = 0;

    switch (version)
    {
        // Tesla
        case 0x10:
        case 0x11:
        case 0x12:
        case 0x13:
            cores = 8; 
            break;

        // Fermi
        case 0x20:
            cores = 32; 
            break;

        case 0x21:
            cores = 48;
            break;

        // Kepler
        case 0x30:
        case 0x32:
        case 0x35:
        case 0x37:
            cores = 192;
            break;

        // Maxwell
        case 0x50:
        case 0x52:
        case 0x53:
            cores = 128;
            break;

        // Pascal
        case 0x60:
            cores = 64;
            break;

        case 0x61:
        case 0x62:
            cores = 128;
            break;

        // Volta
        case 0x70:
        case 0x72:
            cores = 64;
            break;

        // Turing
        case 0x75:
            cores = 64;
            break;
    }

    return cores;
}
 
//================================================================
//
// CudaInitApiThunk::getProperties
//
//================================================================

bool CudaInitApiThunk::getProperties(int32 deviceIndex, GpuProperties& properties, stdNullPars)
{
    stdBegin;

    CUdevice deviceId = 0;
    REQUIRE_CUDA(cuDeviceGet(&deviceId, deviceIndex));

    int multiprocessorCount = 0;
    REQUIRE_CUDA(cuDeviceGetAttribute(&multiprocessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, deviceId));

    int clockRate = 0;
    REQUIRE_CUDA(cuDeviceGetAttribute(&clockRate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, deviceId));

    int ccMajor = 0;
    int ccMinor = 0;
    REQUIRE_CUDA(cuDeviceGetAttribute(&ccMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, deviceId));
    REQUIRE_CUDA(cuDeviceGetAttribute(&ccMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, deviceId));

    properties.gpuHardware = true;
    properties.multiprocessorCount = multiprocessorCount;
    properties.clockRate = float32(clockRate) * 1e3f;

    int coresPerProcessor = getCudaCoresPerMultiprocessor(ccMajor * 0x10 + ccMinor);
    REQUIRE(coresPerProcessor >= 1);

    float32 totalCores = float32(multiprocessorCount) * float32(coresPerProcessor);
    properties.totalThroughput = float32(clockRate) * 1e3f * totalCores;

    //
    //
    //

    int samplerBaseAlignment = 0;
    REQUIRE_CUDA(cuDeviceGetAttribute(&samplerBaseAlignment, CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, deviceId));
    properties.samplerBaseAlignment = samplerBaseAlignment;

    int samplerPitchAlignment = 0;
    REQUIRE_CUDA(cuDeviceGetAttribute(&samplerPitchAlignment, CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT, deviceId));
    properties.samplerRowAlignment = samplerPitchAlignment;

    //
    //
    //

    Point3D<int> maxDeviceGroupCount = point3D(0);
    REQUIRE_CUDA(cuDeviceGetAttribute(&maxDeviceGroupCount.X, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, deviceId));
    REQUIRE_CUDA(cuDeviceGetAttribute(&maxDeviceGroupCount.Y, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, deviceId));
    REQUIRE_CUDA(cuDeviceGetAttribute(&maxDeviceGroupCount.Z, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, deviceId));
    properties.maxGroupCount = convertExact<SpaceU>(maxDeviceGroupCount);

    REQUIRE_CUDA(cuDeviceGetAttribute(&properties.maxThreadCount.X, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, deviceId));
    REQUIRE_CUDA(cuDeviceGetAttribute(&properties.maxThreadCount.Y, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, deviceId));
    REQUIRE_CUDA(cuDeviceGetAttribute(&properties.maxGroupArea, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, deviceId));

    stdEnd;
}

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Context allocation
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// ContextEx
//
// Support concurrent thread access AFTER creation.
// (Basic CUDA handle supports concurrent access)
//
//================================================================

class ContextEx
{

    //----------------------------------------------------------------
    //
    // Creation stage
    //
    //----------------------------------------------------------------

public:

    ~ContextEx() {destroy();}
    stdbool create(int32 deviceIndex, const GpuProperties& gpuProperties, void*& baseContext, stdPars(ErrorLogExKit));
    void destroy();

    //----------------------------------------------------------------
    //
    // Execution stage
    //
    //----------------------------------------------------------------

public:

    inline bool isCreated() const
        {return created;}

public:

    inline Point3D<SpaceU> maxGroupCount() const
        {return gpuProperties.maxGroupCount;}

    inline Point<Space> maxThreadCount() const
        {return gpuProperties.maxThreadCount;}

public:

    CUcontext getBaseContext() const
        {return cuContext;}

    //----------------------------------------------------------------
    //
    // State variables being constant after creation.
    //
    //----------------------------------------------------------------

private:

    bool created = false;
    CUcontext cuContext = 0;
    GpuProperties gpuProperties;

public:

    GpuModuleKeeper moduleKeeper;

};

//================================================================
//
// ContextEx::create
//
//================================================================

stdbool ContextEx::create(int32 deviceIndex, const GpuProperties& gpuProperties, void*& baseContext, stdPars(ErrorLogExKit))
{
    stdBegin;

    destroy();
    baseContext = 0;

    ////

    CUdevice deviceId = 0;
    REQUIRE_CUDA(cuDeviceGet(&deviceId, deviceIndex));

    REQUIRE_CUDA(cuCtxCreate(&cuContext, CU_CTX_SCHED_SPIN, deviceId));
    REMEMBER_CLEANUP1_EX(cuContextCleanup, DEBUG_BREAK_CHECK(cuCtxDestroy(cuContext) == CUDA_SUCCESS), CUcontext, cuContext);

    ////

    cuContextCleanup.cancel();
    baseContext = cuContext;
    this->gpuProperties = gpuProperties;
    created = true;

    stdEnd;
}

//================================================================
//
// ContextEx::destroy
//
//================================================================

void ContextEx::destroy()
{

    gpuProperties.clear();
    moduleKeeper.destroy();

    ////

    if (cuContext != 0)
    {
        DEBUG_BREAK_CHECK(cuCtxDestroy(cuContext) == CUDA_SUCCESS);
        cuContext = 0;
    }

    created = false;
}

//================================================================
//
// uncast
//
//================================================================

inline const ContextEx& uncast(const GpuContext& context)
{
    COMPILE_ASSERT(sizeof(ContextEx*) <= sizeof(GpuContext));
    return ** (const ContextEx**) &context;
}

//================================================================
//
// CudaInitApiThunk::createContext
//
//================================================================

bool CudaInitApiThunk::createContext(int32 deviceIndex, GpuContextOwner& result, void*& baseContext, stdNullPars)
{
    stdBegin;

    result.clear();
    baseContext = 0;

    ////

    GpuProperties tmpProperties;
    require(getProperties(deviceIndex, tmpProperties, stdPass));

    ////

    ContextEx* ctx = new (std::nothrow) ContextEx;
    REQUIRE(ctx != 0);
    REMEMBER_CLEANUP1_EX(contextAllocCleanup, delete ctx, ContextEx*, ctx);

    require(ctx->create(deviceIndex, tmpProperties, baseContext, stdPass));

    ////

    GpuContext tmpContext; // not very elegant, but...
    COMPILE_ASSERT(sizeof(ContextEx*) <= sizeof(tmpContext));
    * (ContextEx**) &tmpContext = ctx;

    ////

    require(ctx->moduleKeeper.create(tmpContext, stdPassKit(kitCombine(kit, getKit()))));
    REMEMBER_CLEANUP1_EX(moduleKeeperCleanup, ctx->moduleKeeper.destroy(), ContextEx*, ctx);

    ////

    GpuContextDeallocContext& deallocContext = result.owner.replace(destroyContext);

    COMPILE_ASSERT(sizeof(ContextEx*) <= sizeof(deallocContext));
    * (ContextEx**) &deallocContext = ctx;

    GpuContext& resultBase = result;
    COMPILE_ASSERT(sizeof(ContextEx*) <= sizeof(resultBase));
    * (ContextEx**) &resultBase = ctx;

    ////

    contextAllocCleanup.cancel();
    moduleKeeperCleanup.cancel();

    ////

    stdEnd;
}

//================================================================
//
// CudaInitApiThunk::destroyContext
//
//================================================================

void CudaInitApiThunk::destroyContext(GpuContextDeallocContext& deallocContext)
{
    COMPILE_ASSERT(sizeof(ContextEx*) <= sizeof(deallocContext));
    ContextEx*& context = * (ContextEx**) &deallocContext;
    delete context;
    context = 0;
}

//================================================================
//
// CudaInitApiThunk::setThreadContext
//
//================================================================

bool CudaInitApiThunk::setThreadContext(const GpuContext& context, stdNullPars)
{
    stdBegin;

    const ContextEx& ctx = uncast(context);

    REQUIRE(ctx.isCreated());
    REQUIRE_CUDA(cuCtxSetCurrent(ctx.getBaseContext()));

    stdEnd;
}

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Module loading
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// CudaInitApiThunk::createModuleFromBinary
//
//================================================================

bool CudaInitApiThunk::createModuleFromBinary(const GpuContext& context, const Array<const uint8>& binary, GpuModuleOwner& result, stdNullPars)
{
    stdBegin;

    result.clear();

    ////

    ARRAY_EXPOSE(binary);

    CUmodule module = 0;
    REQUIRE_CUDA(cuModuleLoadData(&module, binaryPtr));

    ////

    GpuModuleDeallocContext& ownerContext = result.owner.replace(destroyModule);

    COMPILE_ASSERT(sizeof(CUmodule) <= sizeof(ownerContext));
    (CUmodule&) ownerContext = module;

    ////

    GpuModule& resultBase = result;
    COMPILE_ASSERT(sizeof(CUmodule) <= sizeof(resultBase));
    (CUmodule&) resultBase = module;

    ////

    stdEnd;
}

//================================================================
//
// CudaInitApiThunk::destroyModule
//
//================================================================

void CudaInitApiThunk::destroyModule(GpuModuleDeallocContext& deallocContext)
{
    COMPILE_ASSERT(sizeof(CUmodule) <= sizeof(deallocContext));
    CUmodule& context = (CUmodule&) deallocContext;

    CUresult err = cuModuleUnload(context);
    DEBUG_BREAK_CHECK(err == CUDA_SUCCESS);

    context = 0;
}

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Kernel loading
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// CudaInitApiThunk::createKernelFromModule
//
//================================================================

bool CudaInitApiThunk::createKernelFromModule(const GpuModule& module, const char* kernelName, GpuKernelOwner& result, stdNullPars)
{
    stdBegin;

    result.clear();

    ////

    const CUmodule& cuModule = (const CUmodule&) module;

    ////

    CUfunction kernel = 0;
    REQUIRE_CUDA(cuModuleGetFunction(&kernel, cuModule, kernelName));

    ////

    GpuKernel& resultBase = result;
    COMPILE_ASSERT(sizeof(CUfunction) <= sizeof(resultBase));
    (CUfunction&) resultBase = kernel;

    ////

    stdEnd;
}

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Sampler loading
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// CudaInitApiThunk::getSamplerFromModule
//
//================================================================

bool CudaInitApiThunk::getSamplerFromModule(const GpuModule& module, const char* samplerName, GpuSamplerOwner& result, stdNullPars)
{
    stdBegin;

    result.clear();

    ////

    const CUmodule& cuModule = (const CUmodule&) module;

    ////

    CUtexref sampler = 0;

    REQUIRE_CUDA(cuModuleGetTexRef(&sampler, cuModule, samplerName));

    ////

    GpuSampler& resultBase = result;
    COMPILE_ASSERT(sizeof(CUtexref) <= sizeof(resultBase));
    (CUtexref&) resultBase = sampler;

    ////

    stdEnd;
}

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Memory allocation
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// CudaCpuAllocCore
//
//================================================================

struct CudaCpuAllocCore
{
    inline stdbool operator()(CpuAddrU& result, CpuAddrU allocSize, stdPars(CudaInitApiThunkKit))
    {
        stdBegin;

        using AllocFunc = CUresult CUDAAPI (void** pp, size_t bytesize);
        AllocFunc* allocFunc = cuMemAllocHost; // ensure prototype with size_t

        REQUIRE(allocSize <= TYPE_MAX(size_t));

        void* ptr = 0;
        REQUIRE_CUDA(allocFunc(&ptr, size_t(allocSize)));

        COMPILE_ASSERT(sizeof(void*) == sizeof(CpuAddrU));
        result = CpuAddrU(ptr);

        ////

        if (initAllocatedBlocks)
            memset(ptr, 0xCC, allocSize);

        ////

        if (reportAllocatedBlocks)
            printMsg(kit.msgLog, STR("Allocated CPU memory range %0 .. %1"), hex(result), hex(result + allocSize - 1));

        stdEnd;
    }
};

//================================================================
//
// CudaCpuAllocThunk::alloc
//
//================================================================

bool CudaCpuAllocThunk::alloc(const GpuContext& context, CpuAddrU size, CpuAddrU alignment, GpuMemoryOwner& owner, CpuAddrU& result, stdNullPars)
{
    if (size == 0)
        {result = 0; owner.clear(); return true;}

    CudaCpuAllocCore coreAlloc;
    return sysAllocAlignShell<CpuAddrU>(size, alignment, owner, result, coreAlloc, dealloc, stdPassThru);
}

//================================================================
//
// CudaCpuAllocThunk::dealloc
//
//================================================================

void CudaCpuAllocThunk::dealloc(MemoryDeallocContext& deallocContext)
{
    CpuAddrU& memAddr = (CpuAddrU&) deallocContext;

    COMPILE_ASSERT(sizeof(void*) == sizeof(CpuAddrU));
    void* memPtr = (void*) memAddr;

    DEBUG_BREAK_CHECK(cuMemFreeHost(memPtr) == CUDA_SUCCESS);

    memAddr = 0;
}

//================================================================
//
// CudaGpuAllocCore
//
//================================================================

struct CudaGpuAllocCore
{
    inline stdbool operator()(GpuAddrU& result, GpuAddrU allocSize, stdPars(CudaInitApiThunkKit))
    {
        stdBegin;

        using AllocFunc = CUresult CUDAAPI (CUdeviceptr* pp, size_t bytesize);
        AllocFunc* allocFunc = cuMemAlloc; // ensure prototype

        REQUIRE(allocSize <= TYPE_MAX(size_t));

        CUdeviceptr ptr = 0;
        REQUIRE_CUDA(allocFunc(&ptr, size_t(allocSize)));

        COMPILE_ASSERT(sizeof(CUdeviceptr) <= sizeof(GpuAddrU));
        result = GpuAddrU(ptr);

        ////

        if (initAllocatedBlocks)
        {
            REQUIRE(allocSize <= SIZE_MAX); // workaround quirks of CUDA API
            REQUIRE_CUDA(cuMemsetD8(ptr, 0xCC, size_t(allocSize)));
            REQUIRE_CUDA(cuStreamSynchronize(nullptr));
        }

        ////

        if (reportAllocatedBlocks)
            printMsg(kit.msgLog, STR("Allocated GPU memory range %0 .. %1"), hex(result), hex(result + allocSize - 1));

        ////

        stdEnd;
    }
};

//================================================================
//
// CudaGpuAllocThunk::alloc
//
//================================================================

bool CudaGpuAllocThunk::alloc(const GpuContext& context, GpuAddrU size, GpuAddrU alignment, GpuMemoryOwner& owner, GpuAddrU& result, stdNullPars)
{
    if (size == 0)
        {result = 0; owner.clear(); return true;}

    CudaGpuAllocCore coreAlloc;
    return sysAllocAlignShell<GpuAddrU>(size, alignment, owner, result, coreAlloc, dealloc, stdPassThru);
}

//================================================================
//
// CudaGpuAllocThunk::dealloc
//
//================================================================

void CudaGpuAllocThunk::dealloc(MemoryDeallocContext& deallocContext)
{
    GpuAddrU& memAddr = (GpuAddrU&) deallocContext;

    CUdeviceptr memPtr = CUdeviceptr(memAddr);

    CUresult result = cuMemFree(memPtr);
    DEBUG_BREAK_CHECK_CUDA(result);

    memAddr = 0;
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
// CudaInitApiThunk::createTexture
//
//================================================================

bool CudaInitApiThunk::createTexture(const GpuContext& context, const Point<Space>& size, GpuChannelType chanType, int rank, GpuTextureOwner& result, stdNullPars)
{
    stdBegin;

    ++textureAllocCount;

    ////

    result.clear();

    ////

    CUDA_ARRAY_DESCRIPTOR desc;

    REQUIRE(size >= 1);
    desc.Width = size.X;
    desc.Height = size.Y;
    REQUIRE(cudaChannelFormat(chanType, desc.Format));
    desc.NumChannels = rank;

    CUarray texture = 0;
    REQUIRE_CUDA(cuArrayCreate(&texture, &desc));

    ////

    GpuTexture& gpuTexture = result;

    COMPILE_ASSERT(sizeof(CUarray) <= sizeof(GpuTexture));
    (CUarray&) gpuTexture = texture;

    ////

    GpuTextureDeallocContext& ownerContext = result.owner.replace(destroyTexture);

    COMPILE_ASSERT(sizeof(CUarray) <= sizeof(ownerContext));
    (CUarray&) ownerContext = texture;

    ////

    stdEnd;
}

//================================================================
//
// CudaInitApiThunk::destroyTexture
//
//================================================================

void CudaInitApiThunk::destroyTexture(GpuTextureDeallocContext& deallocContext)
{
    COMPILE_ASSERT(sizeof(CUarray) <= sizeof(deallocContext));
    CUarray& texture = (CUarray&) deallocContext;

    CUresult err = cuArrayDestroy(texture);
    DEBUG_BREAK_CHECK(err == CUDA_SUCCESS);

    texture = 0;
}

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Coverage support
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// GpuCoverageKit
//
//================================================================

KIT_COMBINE2(GpuCoverageKit, ErrorLogKit, ErrorLogExKit);

//================================================================
//
// CoverageEvent
//
//================================================================

struct CoverageEvent
{
    GpuEventOwner startEvent;
    GpuEventOwner stopEvent;
};

//================================================================
//
// Because of bad NVIDIA event interface, a context switch can happen between two events,
// resulting in totally wrong kernel time.
// Such events are not random, but hit regular place inside the frame loop, which can be caused
// by repainting or something else.
//
// So, several methods of fighting with it are implemented:
//
// * Coverage multiplier can repeat each kernel call N times, taking the smallest time.
//   But this can result in lowering L2 cache influence (all data is fetched in cache after 1st call).
//
// * Trap events: inserting fake events before and after actual kernel calls.
//   The more trap events are used, the less the probability of a bad measurement is.
//
//================================================================

static const Space coverageMultiplier = 2;
static const Space coverageTotalTrapCount = 0;
static const bool coverageTryToFlushCache = true;

//================================================================
//
// CoverageRecord
//
//================================================================

struct CoverageRecord
{

    //
    // Constant fields (changed only at initialization)
    //

    CoverageEvent events[coverageMultiplier];

    ////

    GpuEventOwner trapEvent;

    //
    // Variable fields
    //

    ProfilerNodeLink profilerNode;

    float32 kernels = 0;
    float32 groups = 0;

};

//================================================================
//
// flushCoverageRecord
//
//================================================================

sysinline stdbool flushCoverageRecord(CoverageRecord& r, bool& syncFlagLatch, Profiler* profiler, stdPars(GpuCoverageKit))
{
    stdBegin;

    if_not (r.profilerNode.connected() && profiler)
        return true;

    ////

    float32 minTimeMs = typeMax<float32>();

    for (Space k = 0; k < coverageMultiplier; ++k)
    {
        CUevent startEvent = uncast(r.events[k].startEvent);
        CUevent stopEvent = uncast(r.events[k].stopEvent);

        CUresult queryResult = cuEventQuery(stopEvent);
        REQUIRE(queryResult == CUDA_SUCCESS || queryResult == CUDA_ERROR_NOT_READY);

        if_not (queryResult == CUDA_SUCCESS)
        {
            syncFlagLatch = true;
            REQUIRE_CUDA(cuEventSynchronize(stopEvent));
        }

        REQUIRE(cuEventQuery(startEvent) == CUDA_SUCCESS);

        ////

        float32 timeMs = 0;
        REQUIRE_CUDA(cuEventElapsedTime(&timeMs, startEvent, stopEvent));
        minTimeMs = minv(minTimeMs, timeMs);
    }

    ////

    float32 deviceTime = 1e-3f * minTimeMs;

    LinearTransform<float32> overheadPredictor = linearTransform(1.98e-6f, 0.0099f); // measured on GTX 780, Win7 64-bit

    float32 predictedOverhead = (overheadPredictor.C0 * r.kernels + overheadPredictor.C1 * r.groups) * 1e-3f;

    profiler->addDeviceTime(r.profilerNode, deviceTime, predictedOverhead);
    r.profilerNode.disconnect();

    ////

    stdEnd;
}

//================================================================
//
// CudaMemoryBlock
//
//================================================================

class CudaMemoryBlock
{

public:

    CudaMemoryBlock() {memPtr = 0; memSize = 0;}
    ~CudaMemoryBlock() {dealloc();}

public:

    bool alloc(GpuAddrU size)
    {
        dealloc();

        using AllocFunc = CUresult CUDAAPI (CUdeviceptr* pp, size_t bytesize);
        AllocFunc* allocFunc = cuMemAlloc; // ensure prototype

        bool ok = true;
        check_flag(size <= SIZE_MAX, ok); // workaround quirks of CUDA API
        if (ok) check_flag(allocFunc(&memPtr, size_t(size)) == CUDA_SUCCESS, ok);

        if (!ok) memPtr = 0;
        if (ok) memSize = size;

        return ok;
    }

public:

    void dealloc()
    {
        if (memPtr)
            {cuMemFree(memPtr); memPtr = 0; memSize = 0;}
    }

    GpuAddrU size() const {return memSize;}

public:

    GpuAddrU getAddr() const {return GpuAddrU(memPtr);}

private:

    CUdeviceptr memPtr;
    GpuAddrU memSize;

};

//================================================================
//
// CoverageQueue
//
//================================================================

class CoverageQueue
{

public:

    KIT_COMBINE2(AllocKit, MallocKit, ErrorLogKit);

public:

    stdbool allocate(Space coverageQueueCapacity, const GpuContext& context, GpuEventAllocator& gpuEventAlloc, stdPars(AllocKit));

    void deallocate() {history.dealloc(); allocated = false;}
    bool isAllocated() const {return allocated;}
    Space allocSize() const {return history.allocSize();}

public:

    stdbool getFreeCoverageSlot(CoverageRecord*& coverageRecord, Profiler* profiler, stdPars(GpuCoverageKit));

    inline void advanceCoverageSlot()
        {history.addAdvance();}

public:

    void discardAll();

    stdbool flushAll(Profiler* profiler, stdPars(GpuCoverageKit));

public:

    bool syncHappened = false;

private:

    bool allocated = false;
    HistoryObj<CoverageRecord> history;

public:

    static const size_t cacheFlushMemSize = 1572864; // 1.5 Mb
    CudaMemoryBlock cacheFlushMemBlock;

};

//================================================================
//
// CoverageQueue::allocate
//
//================================================================

stdbool CoverageQueue::allocate(Space coverageQueueCapacity, const GpuContext& context, GpuEventAllocator& gpuEventAlloc, stdPars(AllocKit))
{
    stdBegin;

    deallocate();

    ////

    CoverageQueue& self = *this;

    require(history.realloc(coverageQueueCapacity, stdPassKit(kitCombine(kit, CpuFastAllocKit(kit.malloc), DataProcessingKit(true)))));
    REMEMBER_CLEANUP1_EX(queueCleanup, self.history.dealloc(), CoverageQueue&, self);

    REQUIRE(cacheFlushMemBlock.alloc(cacheFlushMemSize));
    REMEMBER_CLEANUP1_EX(auxMemCleanup, self.cacheFlushMemBlock.dealloc(), CoverageQueue&, self);

    ////

    for (Space i = 0; i < coverageQueueCapacity; ++i)
    {
        CoverageRecord* r = history.add();
        REQUIRE(r != 0);

        for (Space k = 0; k < coverageMultiplier; ++k)
        {
            require(gpuEventAlloc.createEvent(context, true, r->events[k].startEvent, stdPass));
            require(gpuEventAlloc.createEvent(context, true, r->events[k].stopEvent, stdPass));
        }

        if (coverageTotalTrapCount)
            require(gpuEventAlloc.createEvent(context, false, r->trapEvent, stdPass));
    }

    ////

    history.clear();

    ////

    allocated = true;
    queueCleanup.cancel();
    auxMemCleanup.cancel();

    ////

    stdEnd;
}

//================================================================
//
// CoverageQueue::getFreeCoverageSlot
//
//================================================================

stdbool CoverageQueue::getFreeCoverageSlot(CoverageRecord*& coverageRecord, Profiler* profiler, stdPars(GpuCoverageKit))
{
    stdBegin;

    CoverageRecord* r = history.addLocation();
    REQUIRE(r != 0);

    ////

    require(flushCoverageRecord(*r, syncHappened, profiler, stdPass));

    ////

    coverageRecord = r;

    stdEnd;
}

//================================================================
//
// CoverageQueue::discardAll
//
//================================================================

void CoverageQueue::discardAll()
{
    Space count = history.size();

    for (Space i = count-1; i >= 0; --i)
    {
        CoverageRecord* r = history[i];
        if (r) r->profilerNode.disconnect();
    }

    history.clear();
}

//================================================================
//
// CoverageQueue::flushAll
//
//================================================================

stdbool CoverageQueue::flushAll(Profiler* profiler, stdPars(GpuCoverageKit))
{
    stdBegin;

    ////

    CoverageQueue& self = *this;
    REMEMBER_CLEANUP1(self.discardAll(), CoverageQueue&, self);

    ////

    Space count = history.size();

    for (Space i = count - 1; i >= 0; --i)
    {
        CoverageRecord* r = history[i];
        REQUIRE(r != 0);
        require(flushCoverageRecord(*r, syncHappened, profiler, stdPass));
    }

    ////

    stdEnd;
}

//================================================================
//
// StreamEx
//
//================================================================

struct StreamEx
{

public:

    KIT_COMBINE3(AllocKit, MallocKit, ErrorLogKit, ErrorLogExKit);

    ~StreamEx() {destroy();}

    stdbool create(const GpuContext& context, bool nullStream, void*& baseStream, stdPars(AllocKit));
    void destroy();

public:

    sysinline const GpuContext& getContext()
        {return parentContext;}

public:

    CUstream cuStream = 0;
    GpuContext parentContext;

public:

    CoverageQueue coverageQueue;

};

//================================================================
//
// StreamEx::destroy
//
//================================================================

void StreamEx::destroy()
{
    if (cuStream != 0)
    {
        DEBUG_BREAK_CHECK(cuStreamDestroy(cuStream) == CUDA_SUCCESS);
        cuStream = 0;
    }
}

//================================================================
//
// StreamEx::create
//
//================================================================

stdbool StreamEx::create(const GpuContext& context, bool nullStream, void*& baseStream, stdPars(AllocKit))
{
    stdBegin;

    destroy();
    baseStream = 0;

    ////

    StreamEx& that = *this;

    ////

    if (nullStream)
        cuStream = 0;
    else
        REQUIRE_CUDA(cuStreamCreate(&cuStream, CU_STREAM_NON_BLOCKING));

    ////

    REMEMBER_CLEANUP1_EX(cuStreamCleanup, if (cuStream) DEBUG_BREAK_CHECK(cuStreamDestroy(cuStream) == CUDA_SUCCESS), CUstream, cuStream);

    ////

    parentContext = context;
    cuStreamCleanup.cancel();
    baseStream = cuStream;

    stdEnd;
}

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
// CudaInitApiThunk::coverageInit
//
//================================================================

bool CudaInitApiThunk::coverageInit(const GpuStream& stream, Space coverageQueueCapacity, stdNullPars)
{
    stdBegin;

    REQUIRE(coverageQueueCapacity >= 0);
    StreamEx& streamEx = uncast(stream);

    ////

    CoverageQueue& coverageQueue = streamEx.coverageQueue;

    if_not (coverageQueue.isAllocated() && coverageQueue.allocSize() == coverageQueueCapacity)
        require(coverageQueue.allocate(coverageQueueCapacity, streamEx.getContext(), *this, stdPass));

    ////

    stdEnd;
}

//================================================================
//
// CudaInitThunk::coverageDeinit
//
//================================================================

void CudaInitApiThunk::coverageDeinit(const GpuStream& stream)
{
    uncast(stream).coverageQueue.deallocate();
}

//================================================================
//
// CudaInitApiThunk::coverageGetSyncFlag
// CudaInitApiThunk::coverageClearSyncFlag
//
//================================================================

bool CudaInitApiThunk::coverageGetSyncFlag(const GpuStream& stream)
{
    return uncast(stream).coverageQueue.syncHappened;
}

void CudaInitApiThunk::coverageClearSyncFlag(const GpuStream& stream)
{
    uncast(stream).coverageQueue.syncHappened = false;
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
// CudaInitApiThunk::createStream
//
//================================================================

bool CudaInitApiThunk::createStream(const GpuContext& context, bool nullStream, GpuStreamOwner& result, void*& baseStream, stdNullPars)
{
    stdBegin;

    result.clear();
    baseStream = 0;

    ////

    StreamEx* streamEx = new (std::nothrow) StreamEx;
    REQUIRE(streamEx != 0);
    REMEMBER_CLEANUP1_EX(streamAllocCleanup, delete streamEx, StreamEx*, streamEx);

    require(streamEx->create(context, nullStream, baseStream, stdPass));

    ////

    GpuStreamDeallocContext& deallocContext = result.owner.replace(destroyStream);

    COMPILE_ASSERT(sizeof(StreamEx*) <= sizeof(deallocContext));
    * (StreamEx**) &deallocContext = streamEx;

    GpuStream& gpuStream = result;
    COMPILE_ASSERT(sizeof(StreamEx*) <= sizeof(gpuStream));
    * (StreamEx**) &gpuStream = streamEx;

    ////

    streamAllocCleanup.cancel();

    stdEnd;
}

//================================================================
//
// CudaInitApiThunk::destroyStream
//
//================================================================

void CudaInitApiThunk::destroyStream(GpuStreamDeallocContext& deallocContext)
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
// Benchmarking support
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// callCudaKernel
//
//================================================================

template <typename KernelParams>
stdbool callCudaKernel(const Point3D<Space>& groupCount, const Point3D<Space>& threadCount, const GpuKernelLink& kernelLink, const KernelParams& kernelParams, const GpuStream& stream, stdPars(GpuCoverageKit))
{
    stdBegin;

    StreamEx& streamEx = uncast(stream);
    const ContextEx& ctx = uncast(streamEx.getContext());

    REQUIRE(ctx.isCreated());

    GpuKernel kernelHandle;
    require(ctx.moduleKeeper.fetchKernel(kernelLink, kernelHandle, stdPass));

    ////

    COMPILE_ASSERT(sizeof(CUfunction) <= sizeof(kernelHandle));
    CUfunction cuFunc = (const CUfunction&) kernelHandle;

    CUstream cuStream = uncast(stream).cuStream;

    ////

    size_t paramSize = sizeof(kernelParams);

    void* config[] =
    {
        CU_LAUNCH_PARAM_BUFFER_POINTER, (void*) &kernelParams,
        CU_LAUNCH_PARAM_BUFFER_SIZE, &paramSize,
        CU_LAUNCH_PARAM_END
    };

    REQUIRE(groupCount >= 1);
    REQUIRE(threadCount >= 1);

    REQUIRE_CUDA
    (
        cuLaunchKernel
        (
            cuFunc,
            groupCount.X, groupCount.Y, groupCount.Z,
            threadCount.X, threadCount.Y, threadCount.Z,
            0, // dynamic shared mem bytes
            cuStream,
            NULL,
            config
        )
    );

    stdEnd;
}

//================================================================
//
// callEmptyKernel
//
//================================================================

stdbool callEmptyKernel(const GpuStream& stream, stdPars(GpuCoverageKit))
{
    stdBegin;
    require(callCudaKernel(point3D(1), point3D(1), getEmptyKernelLink(), EmptyKernelParams(), stream, stdPass));
    stdEnd;
}

//================================================================
//
// callReadMemoryKernel
//
//================================================================

stdbool callReadMemoryKernel(const GpuStream& stream, const CudaMemoryBlock& readMemory, stdPars(GpuCoverageKit))
{
    stdBegin;

    GpuAddrU atomCount = readMemory.size() / sizeof(ReadMemoryAtom);

    ReadMemoryKernelParams params;
    params.srcDataPtr = PointerEmulator<GpuAddrU, ReadMemoryAtom>(readMemory.getAddr());
    params.dstDataPtr = PointerEmulator<GpuAddrU, ReadMemoryAtom>(readMemory.getAddr());
    params.atomCount = atomCount;
    params.writeEnabled = false;

    GpuAddrU groupSize = 256;
    GpuAddrU groupCount = divUpNonneg(atomCount, groupSize);

    require(callCudaKernel(point3D(Space(groupCount), 1, 1), point3D(Space(groupSize), 1, 1), getReadMemoryKernelLink(), params, stream, stdPass));

    stdEnd;
}

//================================================================
//
// GPU_COVERAGE_BEGIN
// GPU_COVERAGE_END
//
//================================================================

#define GPU_COVERAGE_BEGIN(kernelsVal, groupsVal) \
    \
    { \
        Profiler* profiler = kit.profiler; \
        StreamEx& streamEx = uncast(stream); \
        CoverageQueue& coverageQueue = streamEx.coverageQueue; \
        bool coverageActive = (gpuCoverageMode == GpuCoverageActive) && profiler && coverageQueue.isAllocated(); \
        \
        CoverageRecord* coverageRec = 0; \
        \
        if (coverageActive) \
        { \
            require(coverageQueue.getFreeCoverageSlot(coverageRec, profiler, stdPass)); \
            coverageRec->kernels = (kernelsVal); \
            coverageRec->groups = (groupsVal); \
        } \
        \
        \
        Space coverageIterations = coverageActive ? coverageMultiplier : 1; \
        \
        for (Space coverageIdx = 0; coverageIdx < coverageIterations; ++coverageIdx) \
        { \
            Space coveragePreTraps = coverageTotalTrapCount / 2; \
            \
            if (coverageIdx != 0 && coverageTryToFlushCache) \
                require(callReadMemoryKernel(stream, coverageQueue.cacheFlushMemBlock, stdPass)); \
            \
            if (coverageActive) \
            { \
                for (Space t = 0; t < coveragePreTraps; ++t) \
                    REQUIRE_CUDA(cuEventRecord(uncast(coverageRec->trapEvent), streamEx.cuStream)); \
                \
                REQUIRE_CUDA(cuEventRecord(uncast(coverageRec->events[coverageIdx].startEvent), streamEx.cuStream)); \
            }


#define GPU_COVERAGE_END \
            \
            if (coverageActive) \
            { \
                REQUIRE_CUDA(cuEventRecord(uncast(coverageRec->events[coverageIdx].stopEvent), streamEx.cuStream)); \
                \
                Space coveragePostTraps = coverageTotalTrapCount - coveragePreTraps; \
                \
                for (Space t = 0; t < coveragePostTraps; ++t) \
                    REQUIRE_CUDA(cuEventRecord(uncast(coverageRec->trapEvent), streamEx.cuStream)); \
            } \
        } \
        \
        if (coverageActive) \
        { \
            profiler->getCurrentNodeLink(*profiler, coverageRec->profilerNode); \
            coverageQueue.advanceCoverageSlot(); \
        } \
    }

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Event allocation
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// CudaInitApiThunk::createEvent
//
//================================================================

bool CudaInitApiThunk::createEvent(const GpuContext& context, bool timingEnabled, GpuEventOwner& result, stdNullPars)
{
    stdBegin;

    result.clear();

    ////

    CUevent event = 0;

    unsigned flags = timingEnabled ? 0 : CU_EVENT_DISABLE_TIMING;
    REQUIRE_CUDA(cuEventCreate(&event, flags));

    ////

    GpuEventDeallocContext& deallocContext = result.owner.replace(destroyEvent);

    COMPILE_ASSERT(sizeof(CUevent) <= sizeof(deallocContext));
    (CUevent&) deallocContext = event;

    ////

    GpuEvent& resultEvent = result;

    COMPILE_ASSERT(sizeof(CUevent) <= sizeof(GpuEvent));
    (CUevent&) resultEvent = event;

    ////

    stdEnd;
}

//================================================================
//
// CudaInitApiThunk::destroyEvent
//
//================================================================

void CudaInitApiThunk::destroyEvent(GpuEventDeallocContext& deallocContext)
{
    CUevent& event = (CUevent&) deallocContext;
    DEBUG_BREAK_CHECK(cuEventDestroy(event) == CUDA_SUCCESS);
    event = 0;
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
// CudaExecApiThunk::copyArrayCpuCpu
// CudaExecApiThunk::copyArrayCpuGpu
// CudaExecApiThunk::copyArrayGpuCpu
// CudaExecApiThunk::copyArrayGpuGpu
//
//================================================================

bool CudaExecApiThunk::copyArrayCpuCpu(CpuAddrU srcPtr, CpuAddrU dstPtr, Space size, const GpuStream& stream, stdNullPars)
{
    stdBegin;
    memcpy((void*) dstPtr, (void*) srcPtr, size);
    stdEnd;
}

bool CudaExecApiThunk::copyArrayCpuGpu(CpuAddrU srcPtr, GpuAddrU dstPtr, Space size, const GpuStream& stream, stdNullPars)
{
    stdBegin;
    GPU_COVERAGE_BEGIN(0, 0);
    
    if (gpuEnqueueMode == GpuEnqueueNormal) 
        REQUIRE_CUDA(cuMemcpyHtoDAsync(CUdeviceptr(dstPtr), (void*) srcPtr, size, uncast(stream).cuStream));

    GPU_COVERAGE_END;
    stdEnd;
}

bool CudaExecApiThunk::copyArrayGpuCpu(GpuAddrU srcPtr, CpuAddrU dstPtr, Space size, const GpuStream& stream, stdNullPars)
{
    stdBegin;
    GPU_COVERAGE_BEGIN(0, 0);

    if (gpuEnqueueMode == GpuEnqueueNormal)
        REQUIRE_CUDA(cuMemcpyDtoHAsync((void*) dstPtr, CUdeviceptr(srcPtr), size, uncast(stream).cuStream));

    GPU_COVERAGE_END;
    stdEnd;
}

bool CudaExecApiThunk::copyArrayGpuGpu(GpuAddrU srcPtr, GpuAddrU dstPtr, Space size, const GpuStream& stream, stdNullPars)
{
    stdBegin;
    GPU_COVERAGE_BEGIN(0, 0);
    
    if (gpuEnqueueMode == GpuEnqueueNormal) 
        REQUIRE_CUDA(cuMemcpyDtoDAsync(CUdeviceptr(dstPtr), CUdeviceptr(srcPtr), size, uncast(stream).cuStream));

    GPU_COVERAGE_END;
    stdEnd;
}

//================================================================
//
// genericMatrixCopy
//
//================================================================

template <typename SrcAddrU, typename DstAddrU>
inline stdbool genericMatrixCopy
(
    CUmemorytype srcType, SrcAddrU srcPtr, Space srcBytePitch,
    CUmemorytype dstType, DstAddrU dstPtr, Space dstBytePitch,
    Space byteSizeX, Space sizeY,
    const GpuStream& stream,
    stdPars(DiagnosticKit)
)
{
    stdBegin;

    CUDA_MEMCPY2D params;

    params.srcXInBytes = 0;
    params.srcY = 0;
    params.srcMemoryType = srcType;
    params.srcHost = (void*) srcPtr;
    params.srcDevice = CUdeviceptr(srcPtr);
    params.srcArray = NULL;
    ASSIGN_CONVERT(params.srcPitch, srcBytePitch);

    params.dstXInBytes = 0;
    params.dstY = 0;
    params.dstMemoryType = dstType;
    params.dstHost = (void*) dstPtr;
    params.dstDevice = CUdeviceptr(dstPtr);
    params.dstArray = NULL;
    ASSIGN_CONVERT(params.dstPitch, dstBytePitch);

    ASSIGN_CONVERT(params.WidthInBytes, byteSizeX);
    ASSIGN_CONVERT(params.Height, sizeY);

    REQUIRE_CUDA(cuMemcpy2DAsync(&params, uncast(stream).cuStream));

    stdEnd;
}

//================================================================
//
// CudaExecApiThunk::copyMatrixCpuCpu
// CudaExecApiThunk::copyMatrixCpuGpu
// CudaExecApiThunk::copyMatrixGpuCpu
// CudaExecApiThunk::copyMatrixGpuGpu
//
//================================================================

#define TMP_MACRO(funcName, SrcAddr, DstAddr, srcType, dstType) \
    \
    bool CudaExecApiThunk::funcName \
    ( \
        SrcAddr srcAddr, Space srcBytePitch, \
        DstAddr dstAddr, Space dstBytePitch, \
        Space byteSizeX, Space sizeY, \
        const GpuStream& stream, \
        stdNullPars \
    ) \
    { \
        stdBegin; \
        \
        GPU_COVERAGE_BEGIN(0, 0); \
        \
        if (gpuEnqueueMode == GpuEnqueueNormal) \
        { \
            require \
            ( \
                genericMatrixCopy \
                ( \
                    srcType, srcAddr, srcBytePitch, \
                    dstType, dstAddr, dstBytePitch, \
                    byteSizeX, sizeY, \
                    stream, \
                    stdPassThru \
                ) \
            ); \
        } \
        \
        GPU_COVERAGE_END; \
        \
        stdEnd; \
    }

TMP_MACRO(copyMatrixCpuCpu, CpuAddrU, CpuAddrU, CU_MEMORYTYPE_HOST, CU_MEMORYTYPE_HOST);
TMP_MACRO(copyMatrixCpuGpu, CpuAddrU, GpuAddrU, CU_MEMORYTYPE_HOST, CU_MEMORYTYPE_DEVICE);
TMP_MACRO(copyMatrixGpuCpu, GpuAddrU, CpuAddrU, CU_MEMORYTYPE_DEVICE, CU_MEMORYTYPE_HOST);
TMP_MACRO(copyMatrixGpuGpu, GpuAddrU, GpuAddrU, CU_MEMORYTYPE_DEVICE, CU_MEMORYTYPE_DEVICE);

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
// cudaAddrMode
//
//================================================================

inline CUaddress_mode cudaAddrMode(BorderMode borderMode)
{
    CUaddress_mode result = CU_TR_ADDRESS_MODE_BORDER;

    if (borderMode == BORDER_CLAMP)
        result = CU_TR_ADDRESS_MODE_CLAMP;

    if (borderMode == BORDER_ZERO)
        result = CU_TR_ADDRESS_MODE_BORDER;

    if (borderMode == BORDER_MIRROR)
        result = CU_TR_ADDRESS_MODE_MIRROR;

    if (borderMode == BORDER_WRAP)
        result = CU_TR_ADDRESS_MODE_WRAP;

    return result;
}

//================================================================
//
// CudaExecApiThunk::setSamplerImage
//
// Overhead is approx 0.7e-6 sec.
//
//================================================================

bool CudaExecApiThunk::setSamplerImage
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
    stdBegin;

    REQUIRE(imageSize >= 0);

    if_not (imageSize >= 1)
        return true;

    //
    // Get texref
    //

    const ContextEx& ctx = uncast(context);
    REQUIRE(ctx.isCreated());

    GpuSampler samplerHandle;
    require(ctx.moduleKeeper.fetchSampler(sampler, samplerHandle, stdPass));

    COMPILE_ASSERT(sizeof(CUtexref) <= sizeof(samplerHandle));
    CUtexref texref = (const CUtexref&) samplerHandle;

    //
    // Check image
    //

    COMPILE_ASSERT(sizeof(Space) <= sizeof(size_t));
    REQUIRE(imageSize.X >= 0 && imageSize.Y >= 0);

    REQUIRE(imageBytePitch >= 0);

    //
    // Set address 2D
    //

    CUDA_ARRAY_DESCRIPTOR desc;

    desc.Width = imageSize.X;
    desc.Height = imageSize.Y;
    REQUIRE(cudaChannelFormat(chanType, desc.Format));
    desc.NumChannels = rank;

    REQUIRE_CUDA(cuTexRefSetAddress2D(texref, &desc, imageBaseAddr, imageBytePitch));

    //
    // Address mode
    //

    CUaddress_mode addressMode = cudaAddrMode(borderMode);

    REQUIRE_CUDA(cuTexRefSetAddressMode(texref, 0, addressMode));
    REQUIRE_CUDA(cuTexRefSetAddressMode(texref, 1, addressMode));

    //
    // Filter mode
    //

    REQUIRE_CUDA(cuTexRefSetFilterMode(texref, linearInterpolation ? CU_TR_FILTER_MODE_LINEAR : CU_TR_FILTER_MODE_POINT));

    //
    // Flags
    //

    uint32 flags = 0;

    if_not (readNormalizedFloat)
        flags |= CU_TRSF_READ_AS_INTEGER;

    if (normalizedCoords)
        flags |= CU_TRSF_NORMALIZED_COORDINATES;

    REQUIRE_CUDA(cuTexRefSetFlags(texref, flags));

    ////

    stdEnd;
}

//================================================================
//
// CudaExecApiThunk::setSamplerArray
//
//================================================================

bool CudaExecApiThunk::setSamplerArray
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
    stdBegin;

    //
    // Get texref
    //

    const ContextEx& ctx = uncast(context);
    REQUIRE(ctx.isCreated());

    GpuSampler samplerHandle;
    require(ctx.moduleKeeper.fetchSampler(sampler, samplerHandle, stdPass));

    COMPILE_ASSERT(sizeof(CUtexref) <= sizeof(samplerHandle));
    CUtexref texref = (const CUtexref&) samplerHandle;

    //
    // Check array
    //

    COMPILE_ASSERT(sizeof(Space) <= sizeof(size_t));

    //
    // Set address
    //

    size_t byteOffset = 0;
    REQUIRE_CUDA(cuTexRefSetAddress(&byteOffset, texref, arrayAddr, arrayByteSize));
    REQUIRE(byteOffset == 0);

    //
    // Set format
    //

    CUarray_format format;
    REQUIRE(cudaChannelFormat(chanType, format));

    REQUIRE_CUDA(cuTexRefSetFormat(texref, format, rank));

    //
    // Address mode
    //

    CUaddress_mode addressMode = cudaAddrMode(borderMode);
    REQUIRE_CUDA(cuTexRefSetAddressMode(texref, 0, addressMode));

    //
    // Filter mode
    //

    REQUIRE_CUDA(cuTexRefSetFilterMode(texref, linearInterpolation ? CU_TR_FILTER_MODE_LINEAR : CU_TR_FILTER_MODE_POINT));

    //
    // Flags
    //

    uint32 flags = 0;

    if_not (readNormalizedFloat)
        flags |= CU_TRSF_READ_AS_INTEGER;

    if (normalizedCoords)
        flags |= CU_TRSF_NORMALIZED_COORDINATES;

    REQUIRE_CUDA(cuTexRefSetFlags(texref, flags));

    ////

    stdEnd;
}

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Kernel calling
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// CudaExecApiThunk::callKernel
//
//================================================================

bool CudaExecApiThunk::callKernel
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
    stdBeginScoped;

    stdEnterElemCount(dbgElemCount);

    ////

    StreamEx& streamEx = uncast(stream);
    const ContextEx& ctx = uncast(streamEx.getContext());

    REQUIRE(ctx.isCreated());

    GpuKernel kernelHandle;
    require(ctx.moduleKeeper.fetchKernel(gpuEnqueueMode == GpuEnqueueNormal ? kernelLink : getEmptyKernelLink(), kernelHandle, stdPass));

    ////

    COMPILE_ASSERT(sizeof(CUfunction) <= sizeof(kernelHandle));
    CUfunction cuFunc = (const CUfunction&) kernelHandle;

    CUstream cuStream = uncast(stream).cuStream;

    ////

    Point3D<SpaceU> maxGroupCount = ctx.maxGroupCount();
    Point<Space> maxThreadCount = ctx.maxThreadCount();

    REQUIRE(convertExact<SpaceU>(groupCount) <= maxGroupCount);
    REQUIRE(convertExact<SpaceU>(threadCount) <= convertExact<SpaceU>(maxThreadCount));

    ////

    GPU_COVERAGE_BEGIN(1.f, float32(groupCount.X) * float32(groupCount.Y) * float32(groupCount.Z));

    ////

    if (allv(groupCount >= 1))
    {

        if (gpuEnqueueMode != GpuEnqueueNormal)
        {
            if (gpuEnqueueMode == GpuEnqueueEmptyKernel)
            {
                EmptyKernelParams emptyKernelParams;
                size_t emptyParamSize = sizeof(emptyKernelParams);

                void* config[] =
                {
                    CU_LAUNCH_PARAM_BUFFER_POINTER, &emptyKernelParams,
                    CU_LAUNCH_PARAM_BUFFER_SIZE, &emptyParamSize,
                    CU_LAUNCH_PARAM_END
                };

                REQUIRE_CUDA
                (
                    cuLaunchKernel
                    (
                        cuFunc,
                        groupCount.X, groupCount.Y, groupCount.Z,
                        threadCount.X, threadCount.Y, 1,
                        0, // dynamic shared mem bytes
                        cuStream,
                        NULL,
                        config
                    )
                );
            }
        }
        else
        {
            void* config[] =
            {
                CU_LAUNCH_PARAM_BUFFER_POINTER, const_cast<void*>(paramPtr),
                CU_LAUNCH_PARAM_BUFFER_SIZE, &paramSize,
                CU_LAUNCH_PARAM_END
            };

            CUresult cudaErr = cuLaunchKernel
            (
                cuFunc,
                groupCount.X, groupCount.Y, groupCount.Z,
                threadCount.X, threadCount.Y, 1,
                0, // shared mem bytes
                cuStream,
                NULL,
                config
            );

            if_not (cudaErr == CUDA_SUCCESS)
            {
                printMsgTrace(kit.errorLogEx, STR("CUDA error: kernel returned %0, groupCount {%1}, threadCount {%2}"),
                    cudaErr, groupCount, threadCount, msgErr, stdPassThru);

                return false;
            }
        }

    }

    ////

    GPU_COVERAGE_END;

    ////

    stdEndScoped;
}

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// StreamEx synchronization
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// CudaExecApiThunk::waitStream
//
//================================================================

bool CudaExecApiThunk::waitStream(const GpuStream& stream, stdNullPars)
{
    stdBegin;

    ////

    StreamEx& streamEx = uncast(stream);

    REQUIRE_CUDA(cuStreamSynchronize(streamEx.cuStream));

    ////

    if (streamEx.coverageQueue.isAllocated())
        require(streamEx.coverageQueue.flushAll(kit.profiler, stdPass));

    ////

    stdEnd;
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

bool CudaExecApiThunk::putEvent(const GpuEvent& event, const GpuStream& stream, stdNullPars)
{
    stdBegin;
    REQUIRE_CUDA(cuEventRecord(uncast(event), uncast(stream).cuStream));
    stdEnd;
}

bool CudaExecApiThunk::putEventDependency(const GpuEvent& event, const GpuStream& stream, stdNullPars)
{
    stdBegin;
    REQUIRE_CUDA(cuStreamWaitEvent(uncast(stream).cuStream, uncast(event), 0));
    stdEnd;
}

bool CudaExecApiThunk::checkEvent(const GpuEvent& event, stdNullPars)
{
    stdBegin;
    CUresult result = cuEventQuery(uncast(event));
    require(result == CUDA_SUCCESS);
    stdEnd;
}

bool CudaExecApiThunk::waitEvent(const GpuEvent& event, bool& realWaitHappened, stdNullPars)
{
    stdBegin;

    realWaitHappened = false;

    if_not (cuEventQuery(uncast(event)) == CUDA_SUCCESS)
    {
        REQUIRE_CUDA(cuEventSynchronize(uncast(event)));
        realWaitHappened = true;
    }

    stdEnd;
}

bool CudaExecApiThunk::eventElapsedTime(const GpuEvent& event1, const GpuEvent& event2, float32& time, stdNullPars)
{
    stdBegin;
    float32 milliseconds = 0;
    REQUIRE_CUDA(cuEventElapsedTime(&milliseconds, uncast(event1), uncast(event2)));
    time = 1e-3f * milliseconds;
    stdEnd;
}

//----------------------------------------------------------------

#endif
