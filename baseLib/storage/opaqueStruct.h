#pragma once

#ifndef HEXLIB_OPAQUE_STRUCT
#define HEXLIB_OPAQUE_STRUCT

#include <type_traits>
#include <cstddef>

//================================================================
//
// OpaqueStruct
//
// The structure with untyped memory of specified size.
// The memory has alignment suitable for any built-in type.
//
//================================================================

template <size_t size>
class OpaqueStruct
{

public:

    template <typename Type>
    inline Type& recast()
    {
        static_assert(sizeof(Self) >= sizeof(Type) && alignof(Self) % alignof(Type) == 0, "");
        return * (Type*) this;
    }

    template <typename Type>
    inline const Type& recast() const
    {
        static_assert(sizeof(Self) >= sizeof(Type) && alignof(Self) % alignof(Type) == 0, "");
        return * (const Type*) this;
    }

private:

    using Self = OpaqueStruct<size>;
    std::aligned_storage_t<size> data; // uses default alignment

};

//================================================================
//
// exchange<OpaqueStruct>
//
//================================================================

template <size_t size>
inline void exchange(OpaqueStruct<size>& a, OpaqueStruct<size>& b)
{
    OpaqueStruct<size> tmp = a;
    a = b;
    b = tmp;
}

//----------------------------------------------------------------

#endif // HEXLIB_OPAQUE_STRUCT
