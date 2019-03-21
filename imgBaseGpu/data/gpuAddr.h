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

    using GpuAddrS = CpuAddrS;
    using GpuAddrU = CpuAddrU;

#else

    #error

#endif
