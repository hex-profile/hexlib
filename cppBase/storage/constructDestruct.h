#pragma once

#include <cstddef>

//================================================================
//
// constructDefault
// constructCopy
// constructParams
// destruct
//
// Explicit calls of constructors and destructor.
//
//================================================================

struct NewLocalizer {};

//----------------------------------------------------------------

inline void* operator new(size_t, NewLocalizer* p)
{
    return p;
}

//----------------------------------------------------------------

inline void operator delete(void*, NewLocalizer*)
{
}

//----------------------------------------------------------------

template <typename Type>
inline void constructDefault(Type& x)
{
    new ((NewLocalizer*) (void*) &x) Type;
}

//----------------------------------------------------------------

template <typename Type>
inline void constructCopy(Type& x, const Type& v)
{
    new ((NewLocalizer*) (void*) &x) Type(v);
}

//================================================================
//
// constructParamsVariadic
//
//================================================================

template <typename Type, typename... Params>
inline void constructParamsVariadic(Type& x, Params&... params)
{
    new ((NewLocalizer*) (void*) &x) Type(params...);
}

//================================================================
//
// constructParams
//
//================================================================

template <typename Type>
inline Type& constructParamsAux__(Type& x)
{
    return x;
}

//----------------------------------------------------------------

#define constructParams(x, ClassName, params) \
    (new ((NewLocalizer*) (void*) &(constructParamsAux__< ClassName >(x))) ClassName params)

//================================================================
//
// destruct
//
//================================================================

template <typename Type>
inline void destruct(Type& x)
{
    x.~Type();
}

//================================================================
//
// valueInit
//
// Assign to default value constructed by value-initialization.
//
//================================================================

template <typename Type>
inline void valueInit(Type& dst)
{
    dst = Type{};
}
