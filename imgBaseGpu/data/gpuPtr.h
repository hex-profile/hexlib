#pragma once

#include "data/pointerEmulator.h"
#include "storage/typeAlignment.h"
#include "data/gpuAddr.h"
#include "dbgptr/dbgptrGate.h"

//================================================================
//
// GpuPtr(Type)
//
// Gives base pointer type (without protected pointers support).
//
//================================================================

#if HEXLIB_PLATFORM == 0

    #define GpuPtr(Type) \
        Type*

#elif HEXLIB_PLATFORM == 1

    #if defined(__CUDA_ARCH__)

        #define GpuPtr(Type) \
            Type*

    #else

        #define GpuPtr(Type) \
            PointerEmulator<GpuAddrU, Type>

    #endif

#else

    #error

#endif

//----------------------------------------------------------------

#define CpuPtr(Type) \
    Type*

//================================================================
//
// CpuPtrType
// GpuPtrType
//
//================================================================

template <typename Type>
struct CpuPtrType
{
    using T = CpuPtr(Type);
};

template <typename Type>
struct GpuPtrType
{
    using T = GpuPtr(Type);
};

//================================================================
//
// Check GpuPtr size/alignment on all devices.
//
// GpuPtr type should have exactly equal size and alignment
// on host and device.
//
//================================================================

COMPILE_ASSERT(sizeof(GpuPtr(int)) == sizeof(GpuAddrU));
COMPILE_ASSERT(alignof(GpuPtr(int)) == sizeof(GpuAddrU));

//================================================================
//
// GpuArrayPtr
// GpuMatrixPtr
//
// Switch to protected pointers (in emulation only).
//
//================================================================

#if HEXLIB_GUARDED_MEMORY

    COMPILE_ASSERT(HEXLIB_PLATFORM == 0);

    #define GpuArrayPtr(Element) \
        ArrayPtr(Element)

    #define GpuArrayPtrCreate(Type, memPtr, memSize, preconditions) \
        ArrayPtrCreate(Type, memPtr, memSize, preconditions)

    #define GpuMatrixPtr(Element) \
        MatrixPtr(Element)

    #define GpuMatrixPtrCreate(Element, memPtr, memPitch, memSizeX, memSizeY, preconditions) \
        MatrixPtrCreate(Element, memPtr, memPitch, memSizeX, memSizeY, preconditions)

#else

    #define GpuArrayPtr(Element) \
        GpuPtr(Element)

    #define GpuArrayPtrCreate(Type, memPtr, memSize, preconditions) \
        (memPtr)

    #define GpuMatrixPtr(Element) \
        GpuPtr(Element)

    #define GpuMatrixPtrCreate(Element, memPtr, memPitch, memSizeX, memSizeY, preconditions) \
        (memPtr)

#endif
