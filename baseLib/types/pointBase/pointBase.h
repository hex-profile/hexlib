#pragma once

#ifndef HEXLIB_POINT_BASE
#define HEXLIB_POINT_BASE

//================================================================
//
// Point
//
// Simple pair of X/Y values.
//
//================================================================

template <typename Type>
struct Point
{
    Type X;
    Type Y;
};

//================================================================
//
// HEXLIB_POINT_INLINE
//
//================================================================

#if defined(__CUDA_ARCH__)
    #define HEXLIB_POINT_INLINE __device__ __host__ inline
    #define HEXLIB_POINT_INLINE_WAS_DEFINED_FOR_DEVICE
#elif defined(_MSC_VER)
    #define HEXLIB_POINT_INLINE __forceinline
#else
    #define HEXLIB_POINT_INLINE inline
#endif

//================================================================
//
// point
//
// point function is used to create Point objects:
//
// point(1.f)
// point(2, 3)
//
//================================================================

template <typename Type>
HEXLIB_POINT_INLINE Point<Type> point(const Type& X, const Type& Y)
{
    Point<Type> result;
    result.X = X;
    result.Y = Y;
    return result;
}

template <typename Type>
HEXLIB_POINT_INLINE Point<Type> point(const Type& value)
{
    Point<Type> result;
    result.X = value;
    result.Y = value;
    return result;
}

//----------------------------------------------------------------

#endif // HEXLIB_POINT_BASE
