#pragma once

#include <stddef.h>

//================================================================
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

//================================================================
//
// constructDefault
// constructDefaultInit
//
//================================================================

template <typename Type>
inline void constructDefault(Type& x)
{
    new ((NewLocalizer*) (void*) &x) Type;
}

//----------------------------------------------------------------

template <typename Type>
inline void constructDefaultInit(Type& x)
{
    new ((NewLocalizer*) (void*) &x) Type{};
}

//================================================================
//
// constructCopy
//
//================================================================

template <typename Type>
inline void constructCopy(Type& x, const Type& v)
{
    new ((NewLocalizer*) (void*) &x) Type(v);
}

//----------------------------------------------------------------

template <typename Type>
inline void constructCopyNoWarning(Type& x, const Type& v)
{

#ifdef __GNUC__
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

    new ((NewLocalizer*) (void*) &x) Type(v);

#ifdef __GNUC__
    #pragma GCC diagnostic pop
#endif

}

//================================================================
//
// constructParamsVariadic
//
//================================================================

template <typename Type, typename... Params>
inline void constructParamsVariadic(Type& x, Params&&... params)
{
    new ((NewLocalizer*) (void*) &x) Type(params...);
}

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
// resetObject
//
//================================================================

template <typename Type>
inline void resetObject(Type& x)
{
    destruct(x);
    constructDefault(x);
}
