#pragma once

#include "compileTools/compileTools.h"

//================================================================
//
// GetSize
//
//================================================================

template <typename Type>
struct GetSize;

#define GET_SIZE_DEFINE(Type, body) \
    \
    struct GetSize<Type> \
    { \
        static sysinline auto func(const Type& value) \
            {return body;} \
    };

//================================================================
//
// equalSize for N arguments
//
//================================================================

template <typename T0>
sysinline bool equalSize(const T0& v0)
{
    return true;
}

template <typename T0, typename... Types>
sysinline bool equalSize(const T0& v0, const Types&... values)
{
    auto size = GetSize<T0>::func(v0);

    bool ok = true;
    int tmp[] = {(ok &= allv(size == GetSize<Types>::func(values)), 0)...};
    return ok;
}

//================================================================
//
// equalLayers for N arguments
//
//================================================================

template <typename T0>
sysinline bool equalLayers(const T0& v0)
{
    return true;
}

template <typename T0, typename... Types>
sysinline bool equalLayers(const T0& v0, const Types&... values)
{
    auto layerCount = getLayerCount(v0);

    bool ok = true;
    int tmp[] = {(ok &= (layerCount == getLayerCount(values)), 0)...};
    return ok;
}
