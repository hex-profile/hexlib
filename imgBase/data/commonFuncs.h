#pragma once

#include "compileTools/compileTools.h"
#include "data/space.h"

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

//----------------------------------------------------------------

template <typename T0, typename... Types>
sysinline bool equalSize(const T0& v0, const Types&... values)
{
    auto size = GetSize<T0>::func(v0);

    bool ok = true;
    char tmp[] = {(ok &= allv(size == GetSize<Types>::func(values)), 'x')...};
    return ok;
}

//================================================================
//
// getLayers
//
//================================================================

template <typename Type>
sysinline Space getLayers(const Type& value);

template <>
sysinline Space getLayers(const Space& layers)
    {return layers;}

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
    auto layers = getLayers(v0);

    bool ok = true;
    char tmp[] = {(ok &= (layers == getLayers(values)), 'x')...};
    return ok;
}
