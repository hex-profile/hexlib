#pragma once

//================================================================
//
// Sampler tools.
//
//================================================================

#if !defined(HEXLIB_PLATFORM) || HEXLIB_PLATFORM == 0
    #include "devSamplerEmu.h"
#elif HEXLIB_PLATFORM == 1
    #include "devSamplerCuda.h"
#else
    #error
#endif
