#include "pixelBuffer.h"

#if HEXLIB_PLATFORM == 1
    #include "gpuLayer/gpuLayerCuda/cudaErrorReport.h"
#endif

#include "errorLog/debugBreak.h"
#include "errorLog/errorLog.h"
#include "numbers/divRoundCompile.h"
#include "storage/rememberCleanup.h"
#include "testbedGL/common/glDebugCheck.h"

//================================================================
//
// PixelBuffer::dealloc
//
//================================================================

void PixelBuffer::dealloc()
{
    if (level == Level::None)
        return;

    //----------------------------------------------------------------
    //
    // Unlock COMPUTE in case of incorrect external usage.
    //
    //----------------------------------------------------------------

    if_not (DEBUG_BREAK_CHECK(level == Level::Allocated))
    {

    #if HEXLIB_PLATFORM == 0

        DEBUG_BREAK_CHECK(glUnmapNamedBuffer(glBuffer) == GL_TRUE);

    #elif HEXLIB_PLATFORM == 1

        DEBUG_BREAK_CHECK_CUDA(cuCtxSynchronize());

        CUstream cuStream = 0;
        DEBUG_BREAK_CHECK_CUDA(cuGraphicsUnmapResources(1, &cudaResource, cuStream));

    #endif
    }

    ////

    DEBUG_BREAK_CHECK_GL(glMakeNamedBufferNonResidentNV(glBuffer));

    ////

#if HEXLIB_PLATFORM == 1

    DEBUG_BREAK_CHECK_CUDA(cuGraphicsUnregisterResource(cudaResource));

#endif

    ////

    DEBUG_BREAK_CHECK_GL(glDeleteBuffers(1, &glBuffer));

    ////

    PixelBufferState& state = *this;
    state = {};
}

//================================================================
//
// PixelBuffer::reallocBase
//
//================================================================

stdbool PixelBuffer::reallocBase(Space elemSize, const Point<Space>& size, Space rowByteAlignment, stdPars(ReportKit))
{
    REQUIRE(level == Level::None || level == Level::Allocated);

    //----------------------------------------------------------------
    //
    // Estimate new size.
    //
    //----------------------------------------------------------------

    REQUIRE(size >= 0);

    ////

    Space pitch{};
    require(getAlignedBufferPitch(elemSize, size.X, rowByteAlignment, pitch, stdPass));

    ////

    Space allocElements{};
    REQUIRE(safeMul(pitch, size.Y, allocElements));

    Space allocBytes{};
    REQUIRE(safeMul(allocElements, elemSize, allocBytes));

    ////

    if (allocBytes == 0)
        allocBytes = 1; // OpenGL does not like zero bytes.

    //----------------------------------------------------------------
    //
    // Try the fast way.
    //
    //----------------------------------------------------------------

    if (level == Level::Allocated && allocBytes <= glAllocBytes)
    {
        currentElemSize = elemSize;
        currentSize = size;
        currentPitch = pitch;

        returnTrue;
    }

    //----------------------------------------------------------------
    //
    // Slow reallocation.
    //
    //----------------------------------------------------------------

    dealloc();

    //----------------------------------------------------------------
    //
    // GL buffer
    //
    //----------------------------------------------------------------

    REQUIRE_GL_FUNC2(glCreateBuffers, glDeleteBuffers);

    REQUIRE_GL(glCreateBuffers(1, &glBuffer));
    REMEMBER_CLEANUP_EX(glBufferCleanup, {DEBUG_BREAK_CHECK_GL(glDeleteBuffers(1, &glBuffer)); glBuffer = 0;});

    ////

    REQUIRE_GL_FUNC(glNamedBufferData);

    REQUIRE_GL(glNamedBufferData(glBuffer, allocBytes, NULL, GL_DYNAMIC_DRAW));

    //----------------------------------------------------------------
    //
    // Register CUDA resource
    //
    //----------------------------------------------------------------

#if HEXLIB_PLATFORM == 1

    REQUIRE_CUDA(cuGraphicsGLRegisterBuffer(&cudaResource, glBuffer, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD));
    REMEMBER_CLEANUP_EX(cudaRegisteringCleanup, {DEBUG_BREAK_CHECK_CUDA(cuGraphicsUnregisterResource(cudaResource)); cudaResource = 0;});

#endif

    //----------------------------------------------------------------
    //
    // Make GL buffer resident and get GL shader address.
    //
    //----------------------------------------------------------------

    REQUIRE_GL_FUNC2(glMakeNamedBufferResidentNV, glMakeNamedBufferNonResidentNV);

    REQUIRE_GL(glMakeNamedBufferResidentNV(glBuffer, GL_READ_WRITE));
    REMEMBER_CLEANUP_EX(glBufferResidenceCleanup, DEBUG_BREAK_CHECK_GL(glMakeNamedBufferNonResidentNV(glBuffer)));

    ////

    REQUIRE_GL_FUNC(glGetNamedBufferParameterui64vNV);

    REQUIRE_GL(glGetNamedBufferParameterui64vNV(glBuffer, GL_BUFFER_GPU_ADDRESS_NV, &glAddress));
    REMEMBER_CLEANUP_EX(glBufferAddressCleanup, glAddress = 0);

    //----------------------------------------------------------------
    //
    // Success.
    //
    //----------------------------------------------------------------

    glBufferCleanup.cancel();
    glBufferResidenceCleanup.cancel();
    glBufferAddressCleanup.cancel();

#if HEXLIB_PLATFORM == 1
    cudaRegisteringCleanup.cancel();
#endif

    ////

    glAllocBytes = allocBytes;

    currentElemSize = elemSize;
    currentSize = size;
    currentPitch = pitch;

    level = Level::Allocated;

    returnTrue;
}

//================================================================
//
// PixelBuffer::lock
//
//================================================================

stdbool PixelBuffer::lock(void* stream, stdPars(ReportKit))
{
    REQUIRE(level == Level::Allocated);

    //----------------------------------------------------------------
    //
    // CPU emulation.
    //
    //----------------------------------------------------------------

#if HEXLIB_PLATFORM == 0

    REQUIRE_GL_FUNC2(glMapNamedBuffer, glUnmapNamedBuffer);

    computeAddress = glMapNamedBuffer(glBuffer, GL_READ_WRITE);
    REQUIRE_GL(true);

    REMEMBER_CLEANUP_EX(lockCleanup, {DEBUG_BREAK_CHECK(glUnmapNamedBuffer(glBuffer) == GL_TRUE); computeAddress = 0;});
    REQUIRE(computeAddress != 0);

#endif

    //----------------------------------------------------------------
    //
    // CUDA.
    //
    //----------------------------------------------------------------

#if HEXLIB_PLATFORM == 1

    auto cuStream = CUstream(stream);

    REQUIRE_CUDA(cuGraphicsMapResources(1, &cudaResource, cuStream));

    REMEMBER_CLEANUP_EX(lockCleanup, DEBUG_BREAK_CHECK_CUDA(cuGraphicsUnmapResources(1, &cudaResource, cuStream)));

    ////

    size_t deviceSize = 0;
    CUdeviceptr devicePtr = 0;
    REQUIRE_CUDA(cuGraphicsResourceGetMappedPointer(&devicePtr, &deviceSize, cudaResource));

    REQUIRE(deviceSize == size_t(glAllocBytes));
    REQUIRE(devicePtr != 0);

    computeAddress = devicePtr;

#endif

    //----------------------------------------------------------------
    //
    // Success.
    //
    //----------------------------------------------------------------

    lockCleanup.cancel();

    level = Level::ComputeLocked;

    returnTrue;
}

//================================================================
//
// PixelBuffer::unlock
//
//================================================================

stdbool PixelBuffer::unlock(void* stream, stdPars(ReportKit))
{
    REQUIRE(level == Level::ComputeLocked);

    //----------------------------------------------------------------
    //
    // Lower the level in any case.
    //
    //----------------------------------------------------------------

    level = Level::Allocated;
    computeAddress = 0;

    //----------------------------------------------------------------
    //
    // CPU emulation.
    //
    //----------------------------------------------------------------

#if HEXLIB_PLATFORM == 0

    REQUIRE(glUnmapNamedBuffer(glBuffer) == GL_TRUE);

#endif

    //----------------------------------------------------------------
    //
    // CUDA.
    //
    //----------------------------------------------------------------

#if HEXLIB_PLATFORM == 1

    auto cuStream = CUstream(stream);

    REMEMBER_CLEANUP_EX
    (
        handleFailure,

        {
            CHECK_CUDA(cuCtxSynchronize());
            CHECK_GL(glFinish());
        }
    );

    ////

    REQUIRE_CUDA(cuGraphicsUnmapResources(1, &cudaResource, cuStream));

    ////

    handleFailure.cancel();

#endif

    ////

    returnTrue;
}
