#pragma once

//----------------------------------------------------------------

#if defined(__CUDA_ARCH__)

    //

#elif defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__) || defined(__arm__) || defined(__aarch64__)

    #include "safeint32_generic.h"
    #define SAFEINT32_SUPPORT

#else

    #error Implementation required

#endif
