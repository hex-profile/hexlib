#include "gpuDevice/devSampler/devSampler.h"

#if HEXLIB_PLATFORM == 1
    #if defined(__CUDA_ARCH__)
        #include "gpuDeviceCuda.h"
    #else
        #include "gpuDeviceNull.h"
    #endif
#else
    #include "gpuDeviceEmu.h"
#endif
