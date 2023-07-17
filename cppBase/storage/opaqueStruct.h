#pragma once

#include "storage/typeAlignment.h"
#include "compileTools/compileTools.h"

//================================================================
//
// OpaqueStruct
//
// The structure with untyped memory of specified size.
// The memory has alignment suitable for any built-in type.
//
// The hash template parameter is introduced to make
// the different types of the same size incompatible.
//
//================================================================

template <size_t size, unsigned hash>
class OpaqueStruct
{

public:

    template <typename Type>
    sysinline Type& recast()
    {
        static_assert(sizeof(Self) >= sizeof(Type) && alignof(Self) % alignof(Type) == 0, "");
        return * (Type*) this;
    }

    template <typename Type>
    sysinline const Type& recast() const
    {
        static_assert(sizeof(Self) >= sizeof(Type) && alignof(Self) % alignof(Type) == 0, "");
        return * (const Type*) this;
    }

private:

    using Self = OpaqueStruct;

    alignas(maxNaturalAlignment) unsigned char data[size];
};

//================================================================
//
// exchange<OpaqueStruct>
//
//================================================================

template <size_t size, unsigned hash>
sysinline void exchange(OpaqueStruct<size, hash>& a, OpaqueStruct<size, hash>& b)
{
    auto tmp = a;
    a = b;
    b = tmp;
}
