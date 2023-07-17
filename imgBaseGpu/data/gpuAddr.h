#pragma once

#include "numbers/int/intBase.h"
#include "storage/addrSpace.h"

//================================================================
//
// GpuAddrS
// GpuAddrU
//
// Should have identical size/alignment on all compilers!
//
//================================================================

#if HEXLIB_PLATFORM == 0

    using GpuAddrS = CpuAddrS;
    using GpuAddrU = CpuAddrU;

#elif HEXLIB_PLATFORM == 1

    #if HEXLIB_GPU_BITNESS == 32

        using GpuAddrS = int32;
        using GpuAddrU = uint32;

    #elif HEXLIB_GPU_BITNESS == 64

        using GpuAddrS = int64;
        using GpuAddrU = uint64;

    #else

        #error

    #endif

#else

    #error

#endif
