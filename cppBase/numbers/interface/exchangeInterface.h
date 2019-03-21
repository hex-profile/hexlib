#pragma once

#include "compileTools/compileTools.h"

//================================================================
//
// exchange
//
// Exchanges two values.
//
//================================================================

template <typename Type>
struct ExchangeCategory;

////

template <typename Category>
struct ExchangeImpl;

////

template <typename Type>
sysinline void exchange(Type& A, Type& B)
    {ExchangeImpl<typename ExchangeCategory<Type>::T>::func(A, B);}

////

struct ExchangeSimple;

////

template <>
struct ExchangeImpl<ExchangeSimple>
{
    template <typename Type>
    static sysinline void func(Type& A, Type& B)
    {
        Type temp = A;
        A = B;
        B = temp;
    }
};

////

template <typename Type>
struct ExchangeCategory<Type*>
{
    using T = ExchangeSimple;
};

////

#define EXCHANGE_DEFINE_SIMPLE(Type, _) \
    template <> \
    struct ExchangeCategory<Type> \
        {using T = ExchangeSimple;};

//================================================================
//
// exchangeByCopying
//
//================================================================

template <typename Type>
static sysinline void exchangeByCopying(Type& A, Type& B)
{
    Type temp = A;
    A = B;
    B = temp;
}
