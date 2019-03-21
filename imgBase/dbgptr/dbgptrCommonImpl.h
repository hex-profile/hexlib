#pragma once

//================================================================
//
// DBGPTR_ERROR_BREAK
//
//================================================================

#if defined(__CUDA_ARCH__)

    #define DBGPTR_ERROR_BREAK()

#elif defined(_MSC_VER)

    #define DBGPTR_ERROR_BREAK() \
        \
        __try \
        { \
            __debugbreak(); \
        } \
        __except (1) \
        { \
        }

#elif defined(__GNUC__)

    #define DBGPTR_ERROR_BREAK() \
        __builtin_trap()

#else

    #error Implement

#endif
