#pragma once

#include "compileTools/compileTools.h"

//================================================================
//
// loadViaSamplerCache
//
//================================================================

template <typename Type>
sysinline Type loadViaSamplerCache(const Type* ptr);

//----------------------------------------------------------------

#if !(defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350))

    template <typename Type>
    sysinline Type loadViaSamplerCache(const Type* ptr)
        {return *ptr;}

#else

    ////

    template <int elemSize>
    struct LoadViaSamplerCacheAux;

    #define TMP_MACRO(ReadType) \
        \
        template <> \
        struct LoadViaSamplerCacheAux<sizeof(ReadType)> \
        { \
            template <typename Type> \
            struct Inner  \
            { \
                COMPILE_ASSERT(sizeof(Type) == sizeof(ReadType) && \
                    alignof(Type) == alignof(ReadType)); \
                \
                static sysinline void func(const Type* ptr, Type& result) \
                    {* (ReadType*) &result = __ldg((const ReadType*) ptr);} \
            }; \
        };

    TMP_MACRO(uint8)
    TMP_MACRO(uint16)
    TMP_MACRO(uint32)
    TMP_MACRO(uint64)

    #undef TMP_MACRO

    ////

    template <typename Type>
    sysinline Type loadViaSamplerCache(const Type* ptr)
    {
        Type result;
        LoadViaSamplerCacheAux<sizeof(Type)>::Inner<Type>::func(ptr, result);
        return result;
    }

#endif

//================================================================
//
// LoadViaSamplerCache
// load mode
//
//================================================================

struct LoadViaSamplerCache
{
    template <typename Type>
    static sysinline Type func(const Type* ptr)
        {return loadViaSamplerCache(ptr);}
};
