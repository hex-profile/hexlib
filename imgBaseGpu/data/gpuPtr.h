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

    #define GpuPtrDistinctType 0

#elif HEXLIB_PLATFORM == 1

    #if defined(__CUDA_ARCH__)

        #define GpuPtr(Type) \
            Type*

        #define GpuPtrDistinctType 0

    #else

        #define GpuPtr(Type) \
            PointerEmulator<GpuAddrU, Type>

        #define GpuPtrDistinctType 1

    #endif

#else

    #error

#endif

//================================================================
//
// CpuPtr
//
//================================================================

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

COMPILE_ASSERT_EQUAL_LAYOUT(GpuPtr(int), GpuAddrU);

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

    #define GpuMatrixPtr(Element) \
        MatrixPtr(Element)

#else

    #define GpuArrayPtr(Element) \
        GpuPtr(Element)

    #define GpuMatrixPtr(Element) \
        GpuPtr(Element)

#endif
