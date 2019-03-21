#if defined(__CUDA_ARCH__)
    # include "halfConvertCuda.h"
#elif defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__) || defined(__arm__) || defined(__aarch64__) // IEEE FP32
    # include "halfConvertEmu.h"
#else
    #error Need to implement
#endif
