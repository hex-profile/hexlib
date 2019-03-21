#pragma once

#include "compileTools/compileTools.h"

//================================================================
//
// Pointer inteface.
//
// Unified access to pointer type, required to support
// pointers to foreign address space.
//
//================================================================

//================================================================
//
// PtrElemType
//
//================================================================

template <typename Pointer>
struct PtrElemType
{
    using T = void;
};

//----------------------------------------------------------------

template <typename Type>
struct PtrElemType<Type*>
{
    using T = Type;
};

//================================================================
//
// PtrAddrType
//
// Corresponding address space types.
//
//================================================================

template <typename Pointer>
struct PtrAddrType
{
    using AddrU = void;
};

//----------------------------------------------------------------

template <typename Type>
struct PtrAddrType<Type*>
{
    COMPILE_ASSERT(sizeof(size_t) == sizeof(Type*));
    using AddrU = size_t;
};
