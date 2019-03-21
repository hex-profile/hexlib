#pragma once

#include "numbers/int/intBase.h"
#include "numbers/int/intType.h"
#include "data/pointerInterface.h"

//================================================================
//
// PointerEmulator
//
// Emulates pointer type arithmetic,
// to use for alien address space pointers.
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

    PointerEmulator() {addr = 0xBAADF00D;}

    //
    // Convert from address type
    //

    inline explicit PointerEmulator(AddrU addr)
        : addr(addr) {}

    //
    // Export to address type
    //

    inline operator AddrU() const
        {return addr;}

    //
    // Export const pointer (fast reinterpret)
    //

public:

    using SelfConstType = PointerEmulator<AddrU, const Type>;

    inline operator const SelfConstType& ()
    {
        COMPILE_ASSERT(sizeof(SelfConstType) == sizeof(Self));
        return * (const SelfConstType*) this;
    }

    //
    // Comparisons
    //

public:

    #define TMP_MACRO(OP) \
        inline bool operator OP(const Self& that) const \
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

    inline Self& operator ++()
        {addr += sizeof(Type); return *this;}

    inline Self& operator --()
        {addr -= sizeof(Type); return *this;}

    inline Self operator ++(int)
        {Self tmp(*this); ++(*this); return tmp;}

    inline Self operator --(int)
        {Self tmp(*this); --(*this); return tmp;}

    //
    // +=
    // -=
    //

public:

    #define TMP_MACRO(AddrS) \
        \
        inline Self& operator +=(AddrS offs) \
        { \
            addr += offs * AddrS(sizeof(Type)); \
            return *this; \
        } \
        \
        inline Self& operator -=(AddrS offs) \
        { \
            addr -= offs * AddrS(sizeof(Type)); \
            return *this; \
        } \
        \
        inline Self operator +(AddrS offs) const \
        { \
            AddrU newAddr = addr + offs * AddrS(sizeof(Type)); \
            return Self(newAddr); \
        } \
        \
        inline Self operator -(AddrS offs) const \
        { \
            AddrU newAddr = addr - offs * AddrS(sizeof(Type)); \
            return Self(newAddr); \
        }

    TMP_MACRO(AddrS)
    // TMP_MACRO(Space)

    #undef TMP_MACRO

    //
    // Operator -(p)
    //

public:

    inline AddrS operator -(const Self& that) const
    {
        AddrS diff = AddrS(this->addr - that.addr); // ring arithmetic
        return diff / AddrS(sizeof(Type)); // signed div
    }

};

//----------------------------------------------------------------

template <typename AddrU, typename Type>
struct ExchangeCategory< PointerEmulator<AddrU, Type> > {using T = ExchangeSimple;};

//================================================================
//
// PtrElemType
//
//================================================================

template <typename Addr, typename Type>
struct PtrElemType< PointerEmulator<Addr, Type> >
{
    using T = Type;
};

//================================================================
//
// PtrAddrType
//
//================================================================

template <typename Addr, typename Type>
struct PtrAddrType< PointerEmulator<Addr, Type> >
{
    using AddrU = Addr;
};
