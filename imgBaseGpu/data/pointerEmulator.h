#pragma once

#include "numbers/int/intBase.h"
#include "numbers/int/intType.h"
#include "data/pointerInterface.h"

//================================================================
//
// PointerEmulator
//
// Emulates pointer type arithmetic for use with pointers
// from foreign address spaces.
//
//================================================================

template <typename AddrU, typename Type>
class PointerEmulator
{

private:

    AddrU addr;

private:

    COMPILE_ASSERT(!TYPE_IS_SIGNED(AddrU));
    using AddrS = TYPE_MAKE_SIGNED(AddrU);

    using Self = PointerEmulator<AddrU, Type>;

public:

    PointerEmulator()
        {addr = 0xBAADF00D;}

    //
    // Convert from address type
    //

    sysinline explicit PointerEmulator(AddrU addr)
        : addr(addr) {}

    //
    // Export to address type
    //

    sysinline operator AddrU() const
        {return addr;}

    //
    // Export const pointer (fast reinterpret)
    //

public:

    using SelfConstType = PointerEmulator<AddrU, const Type>;

    sysinline operator const SelfConstType& () const
    {
        return recastEqualLayout<const SelfConstType>(*this);
    }

    //
    // Comparisons
    //

public:

    #define TMP_MACRO(OP) \
        sysinline bool operator OP(const Self& that) const \
            {return this->addr OP that.addr;}

    TMP_MACRO(>)
    TMP_MACRO(<)
    TMP_MACRO(>=)
    TMP_MACRO(<=)
    TMP_MACRO(==)
    TMP_MACRO(!=)

    #undef TMP_MACRO

    //
    // ++
    // --
    //

public:

    sysinline Self& operator ++()
        {addr += sizeof(Type); return *this;}

    sysinline Self& operator --()
        {addr -= sizeof(Type); return *this;}

    sysinline Self operator ++(int)
        {Self tmp(*this); ++(*this); return tmp;}

    sysinline Self operator --(int)
        {Self tmp(*this); --(*this); return tmp;}

    //
    // +=
    // -=
    //

private:

    template <typename Ofs>
    struct CheckOfsType
    {
        static constexpr bool value = TYPE_IS_BUILTIN_INT(Ofs) && (sizeof(Ofs) <= sizeof(AddrS));
    };

public:

    template <typename Ofs>
    sysinline Self& operator +=(Ofs ofs)
    {
        COMPILE_ASSERT(CheckOfsType<Ofs>::value);

        addr += AddrS(ofs) * AddrS(sizeof(Type));
        return *this;
    }

    template <typename Ofs>
    sysinline Self& operator -=(Ofs ofs)
    {
        COMPILE_ASSERT(CheckOfsType<Ofs>::value);

        addr -= AddrS(ofs) * AddrS(sizeof(Type));
        return *this;
    }

    template <typename Ofs>
    sysinline Self operator +(Ofs ofs) const
    {
        COMPILE_ASSERT(CheckOfsType<Ofs>::value);

        AddrU newAddr = addr + AddrS(ofs) * AddrS(sizeof(Type));
        return Self(newAddr);
    }

    template <typename Ofs>
    sysinline Self operator -(Ofs ofs) const
    {
        COMPILE_ASSERT(CheckOfsType<Ofs>::value);

        AddrU newAddr = addr - AddrS(ofs) * AddrS(sizeof(Type));
        return Self(newAddr);
    }

    //
    // Operator -(p)
    //

public:

    sysinline AddrS operator -(const Self& that) const
    {
        AddrS diff = AddrS(this->addr - that.addr); // ring arithmetic
        return diff / AddrS(sizeof(Type)); // signed div
    }

public:

    sysinline void addByteOffset(AddrS ofs)
    {
        addr += ofs;
    }

};

//----------------------------------------------------------------

template <typename AddrU, typename Type>
struct ExchangeCategory<PointerEmulator<AddrU, Type>> {using T = ExchangeSimple;};

//================================================================
//
// PtrElemType
//
//================================================================

template <typename Addr, typename Type>
struct PtrElemType<PointerEmulator<Addr, Type>>
{
    using T = Type;
};

//================================================================
//
// PtrAddrType
//
//================================================================

template <typename Addr, typename Type>
struct PtrAddrType<PointerEmulator<Addr, Type>>
{
    using AddrU = Addr;
};

//================================================================
//
// PtrRebaseType
//
//================================================================

template <typename Src, typename Addr, typename Dst>
struct PtrRebaseType<PointerEmulator<Addr, Src>, Dst>
{
    using T = PointerEmulator<Addr, Dst>;
};

//================================================================
//
// addOffset
//
//================================================================

template <typename AddrU, typename Type>
sysinline auto addOffset(const PointerEmulator<AddrU, Type>& ptr, TYPE_MAKE_SIGNED(AddrU) ofs)
{
    auto result = ptr;
    result.addByteOffset(ofs);
    return result;
}
