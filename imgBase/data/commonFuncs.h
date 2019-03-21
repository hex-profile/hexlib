#pragma once

#include "compileTools/compileTools.h"

//================================================================
//
// equalSize
//
//================================================================

template <typename Type>
sysinline bool equalSize(const Type& A)
{
    return true;
}

//================================================================
//
// equalSize
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


//----------------------------------------------------------------

template <typename TypeA, typename TypeB>
sysinline bool equalSize(const TypeA& A, const TypeB& B)
{
    return allv(GetSize<TypeA>::func(A) == GetSize<TypeB>::func(B));
}

//================================================================
//
// equalSize for N arguments
//
//================================================================

template <typename T0, typename T1, typename T2>
sysinline bool equalSize(const T0& v0, const T1& v1, const T2& v2)
    {return equalSize(v0, v1) && equalSize(v0, v2);}

template <typename T0, typename T1, typename T2, typename T3>
sysinline bool equalSize(const T0& v0, const T1& v1, const T2& v2, const T3& v3)
    {return equalSize(v0, v1) && equalSize(v0, v2) && equalSize(v0, v3);}

template <typename T0, typename T1, typename T2, typename T3, typename T4>
static sysinline bool equalSize(const T0& v0, const T1& v1, const T2& v2, const T3& v3, const T4& v4)
    {return equalSize(v0, v1) && equalSize(v0, v2) && equalSize(v0, v3) && equalSize(v0, v4);}

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5>
sysinline bool equalSize(const T0& v0, const T1& v1, const T2& v2, const T3& v3, const T4& v4, const T5& v5)
    {return equalSize(v0, v1) && equalSize(v0, v2) && equalSize(v0, v3) && equalSize(v0, v4) && equalSize(v0, v5);}

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
sysinline bool equalSize(const T0& v0, const T1& v1, const T2& v2, const T3& v3, const T4& v4, const T5& v5, const T6& v6)
    {return equalSize(v0, v1) && equalSize(v0, v2) && equalSize(v0, v3) && equalSize(v0, v4) && equalSize(v0, v5) && equalSize(v0, v6);}

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
sysinline bool equalSize(const T0& v0, const T1& v1, const T2& v2, const T3& v3, const T4& v4, const T5& v5, const T6& v6, const T7& v7)
    {return equalSize(v0, v1) && equalSize(v0, v2) && equalSize(v0, v3) && equalSize(v0, v4) && equalSize(v0, v5) && equalSize(v0, v6) && equalSize(v0, v7);}

//================================================================
//
// equalLayers
//
//================================================================

template <typename TypeA, typename TypeB>
sysinline bool equalLayers(const TypeA& A, const TypeB& B)
{
    return getLayerCount(A) == getLayerCount(B);
}

//================================================================
//
// equalLayers for N arguments
//
//================================================================

template <typename T0, typename T1, typename T2>
sysinline bool equalLayers(const T0& v0, const T1& v1, const T2& v2)
    {return equalLayers(v0, v1) && equalLayers(v0, v2);}

template <typename T0, typename T1, typename T2, typename T3>
sysinline bool equalLayers(const T0& v0, const T1& v1, const T2& v2, const T3& v3)
    {return equalLayers(v0, v1) && equalLayers(v0, v2) && equalLayers(v0, v3);}

template <typename T0, typename T1, typename T2, typename T3, typename T4>
static sysinline bool equalLayers(const T0& v0, const T1& v1, const T2& v2, const T3& v3, const T4& v4)
    {return equalLayers(v0, v1) && equalLayers(v0, v2) && equalLayers(v0, v3) && equalLayers(v0, v4);}

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5>
sysinline bool equalLayers(const T0& v0, const T1& v1, const T2& v2, const T3& v3, const T4& v4, const T5& v5)
    {return equalLayers(v0, v1) && equalLayers(v0, v2) && equalLayers(v0, v3) && equalLayers(v0, v4) && equalLayers(v0, v5);}

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
sysinline bool equalLayers(const T0& v0, const T1& v1, const T2& v2, const T3& v3, const T4& v4, const T5& v5, const T6& v6)
    {return equalLayers(v0, v1) && equalLayers(v0, v2) && equalLayers(v0, v3) && equalLayers(v0, v4) && equalLayers(v0, v5) && equalLayers(v0, v6);}

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
sysinline bool equalLayers(const T0& v0, const T1& v1, const T2& v2, const T3& v3, const T4& v4, const T5& v5, const T6& v6, const T7& v7)
    {return equalLayers(v0, v1) && equalLayers(v0, v2) && equalLayers(v0, v3) && equalLayers(v0, v4) && equalLayers(v0, v5) && equalLayers(v0, v6) && equalLayers(v0, v7);}
