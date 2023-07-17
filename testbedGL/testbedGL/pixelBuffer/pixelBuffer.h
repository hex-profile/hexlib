#pragma once

#include <GL/glew.h>

#if HEXLIB_PLATFORM == 1
    #include <cuda.h>
    #include "testbedGL/pixelBuffer/cudaGL.h"
#endif

#include "data/gpuMatrix.h"
#include "errorLog/errorLogKit.h"
#include "stdFunc/stdFunc.h"
#include "lib/imageTools/getAlignedBufferPitch.h"
#include "userOutput/msgLogExKit.h"

//================================================================
//
// Check platforms.
//
//================================================================

#if !(HEXLIB_PLATFORM == 0 || HEXLIB_PLATFORM == 1)
    #error Need to extend support
#endif

//================================================================
//
// PixelBufferState
//
//================================================================

struct PixelBufferState
{
    enum class Level {None, Allocated, ComputeLocked};
    Level level = Level::None;

    ////

    Space currentElemSize = 0;
    Point<Space> currentSize = point(0);
    Space currentPitch = 0;

    ////

    GLuint glBuffer = 0;
    GLuint64EXT glAddress = 0;
    Space glAllocBytes = 0;

    ////

#if HEXLIB_PLATFORM == 0

    void* computeAddress = nullptr;

#elif HEXLIB_PLATFORM == 1

    CUgraphicsResource cudaResource = 0;
    CUdeviceptr computeAddress = 0;

#endif

};

//================================================================
//
// PixelBuffer
//
// Allocates GL buffer, registers in COMPUTE API.
//
//================================================================

class PixelBuffer : private PixelBufferState
{

public:

    inline ~PixelBuffer() {dealloc();}

    using ReportKit = KitCombine<ErrorLogKit, MsgLogExKit>;

    //----------------------------------------------------------------
    //
    // Allocation.
    //
    //----------------------------------------------------------------

    void dealloc();

    ////

    template <typename Element>
    stdbool realloc(const Point<Space>& size, Space rowByteAlignment, stdPars(ReportKit))
        {return reallocBase(sizeof(Element), size, rowByteAlignment, stdPassThru);}

    ////

    stdbool reallocBase(Space elemSize, const Point<Space>& size, Space rowByteAlignment, stdPars(ReportKit));

    //----------------------------------------------------------------
    //
    // COMPUTE locking.
    //
    //----------------------------------------------------------------

    stdbool lock(void* stream, stdPars(ReportKit));

    stdbool unlock(void* stream, stdPars(ReportKit));

    //----------------------------------------------------------------
    //
    // getComputeBuffer
    //
    //----------------------------------------------------------------

    template <typename Element, typename Kit>
    inline stdbool getComputeBuffer(GpuMatrix<Element>& result, stdPars(Kit)) const
    {
        REQUIRE(level == Level::ComputeLocked);

        REQUIRE(sizeof(Element) == currentElemSize);

        using GpuElementPtr = GpuPtr(Element);
        REQUIRE(result.assign(GpuElementPtr(computeAddress), currentPitch, currentSize.X, currentSize.Y));

        returnTrue;
    }

    //----------------------------------------------------------------
    //
    // getGraphicsBuffer
    //
    //----------------------------------------------------------------

    template <typename Kit>
    stdbool getGraphicsBuffer(GLuint64EXT& memPtr, Space& memPitch, stdPars(Kit)) const
    {
        REQUIRE(level == Level::Allocated);

        memPtr = glAddress;
        memPitch = currentPitch;
        returnTrue;
    }

    //----------------------------------------------------------------
    //
    // properties
    //
    //----------------------------------------------------------------

    inline Point<Space> size() const {return currentSize;}

    //----------------------------------------------------------------
    //
    // exchange
    //
    //----------------------------------------------------------------

    inline friend void exchange(PixelBuffer& a, PixelBuffer& b)
    {
        PixelBufferState& aState = a;
        PixelBufferState& bState = b;

        exchangeByCopying(aState, bState);
    }

};
