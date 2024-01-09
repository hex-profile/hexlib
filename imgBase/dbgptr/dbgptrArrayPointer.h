#pragma once

#include "dbgptr/dbgptrCommon.h"
#include "dbgptr/dbgptrReference.h"
#include "data/pointerInterface.h"

//================================================================
//
// DebugArrayPointerByteEngine
//
//================================================================

class DebugArrayPointerByteEngine
{

    friend class DebugMatrixPointerByteEngine;

    //
    // Create empty
    //

public:

    sysinline explicit DebugArrayPointerByteEngine()
    {
        memoryStart = 0;
        memorySize = 0;
        currentPtr = 0;
    }

    //
    // Create from byte array
    //

public:

    sysinline explicit DebugArrayPointerByteEngine(DbgptrAddrU arrPtr, Space arrSize)
    {
        setup(arrPtr, arrSize, arrPtr);
    }

    ////

    void setup(DbgptrAddrU arrPtr, Space arrSize, DbgptrAddrU ptr)
    {
        if_not (arrSize >= 0)
            arrSize = 0;

        memoryStart = arrPtr;
        memorySize = arrSize;
        currentPtr = ptr;
    }

    //
    // Construct / copy
    //

public:

    sysinline explicit DebugArrayPointerByteEngine(const DebugArrayPointerByteEngine& that)
    {
        this->memoryStart = that.memoryStart;
        this->memorySize = that.memorySize;
        this->currentPtr = that.currentPtr;
    }

public:

    sysinline DebugArrayPointerByteEngine& operator =(const DebugArrayPointerByteEngine& that)
    {
        this->memoryStart = that.memoryStart;
        this->memorySize = that.memorySize;
        this->currentPtr = that.currentPtr;

        return *this;
    }

    //
    // Check single byte. The most efficient check.
    //

public:

    sysinline void validateSingleByte() const
    {
        if_not (currentPtr - memoryStart < memorySize)
            errorBreak();
    }

    //
    // Validate range access.
    //

public:

    sysinline void validateArray(bool ok, DbgptrAddrU testSize) const
    {
        check_flag(testSize <= memorySize, ok);
        check_flag(currentPtr - memoryStart <= memorySize - testSize, ok);

        if_not (ok)
            errorBreak();
    }

public:

    void errorBreak() const;

private:

    // The memory beginning.
    DbgptrAddrU memoryStart;

    // The memory size in bytes, >= 0.
    DbgptrAddrU memorySize;

public:

    // Current byte pointer.
    DbgptrAddrU currentPtr;

};

//================================================================
//
// DbgptrArrayPreconditions
//
// PRECONDITIONS:
//
// size >= 0
// size * sizeof(Element) fits into Space type.
//
//================================================================

struct DbgptrArrayPreconditions {};

//================================================================
//
// DebugArrayPointer
//
// The class is based on DebugArrayPointerByteEngine.
//
// When checking for a single element, validateSingleByte is used.
//
// This is identical to checking the whole element, because
// * memory size is multiple of element size E AND
// * (base address mod E) == (current pointer mod E).
//
// This requires ALL modifications of the current pointer to be multiples of element size!
//
//================================================================

template <typename Type>
class DebugArrayPointer
{

    template <typename Any>
    friend class DebugArrayPointer;

    using Self = DebugArrayPointer<Type>;
    using Base = DebugArrayPointerByteEngine;

public:

    using Element = Type;

public:

    //
    // Create empty
    //

    sysinline DebugArrayPointer()
        : base() {}

    struct Null;

    sysinline DebugArrayPointer(Null*)
        : base() {}

    //
    // Basic copy
    //

    sysinline DebugArrayPointer(const Self& that)
        : base(that.base) {}

    sysinline DebugArrayPointer(const Base& base)
        : base(base) {}

    sysinline Self& operator =(const Self& that)
    {
        base = that.base;
        return *this;
    }

    //
    // Advanced copy
    //

    template <typename Other>
    sysinline DebugArrayPointer(const DebugArrayPointer<Other>& that)
        :
        base(that.base)
    {
        Element* checkConversion = (Other*) 0; checkConversion = checkConversion;
    }

    template <typename Other>
    sysinline Self& operator =(const DebugArrayPointer<Other>& that)
    {
        Element* checkConversion = (Other*) 0; checkConversion = checkConversion;

        base = that.base;
        return *this;
    }

    //
    // Array constructor
    //

private:

    static const Space maxSize = spaceMax / sizeof(Element);

public:

    sysinline DebugArrayPointer(Element* arrPtr, Space arrSize, const DbgptrArrayPreconditions&)
    {
        if_not (SpaceU(arrSize) <= SpaceU(maxSize))
            arrSize = 0;

        base.setup(DbgptrAddrU(arrPtr), arrSize * Space(sizeof(Element)), DbgptrAddrU(arrPtr));
    }

    //
    // Base operations
    //

public:

    sysinline const Element& read() const
    {
        base.validateSingleByte();
        return * (Element*) base.currentPtr;
    }

    template <typename Value>
    sysinline void write(const Value& value) const
    {
        base.validateSingleByte();
        * (Element*) base.currentPtr = value;
    }

    sysinline Element& modify() const
    {
        base.validateSingleByte();
        return * (Element*) base.currentPtr;
    }

    //
    // Internal pointer access
    //

public:

    sysinline Element* getPtrForInternalUsageOnly() const
        {return (Element*) base.currentPtr;}

    //
    // ++
    // --
    //

public:

    sysinline Self& operator ++()
        {base.currentPtr += sizeof(Element); return *this;}

    sysinline Self& operator --()
        {base.currentPtr -= sizeof(Element); return *this;}

    sysinline Self operator ++(int)
        {Self tmp(*this); ++(*this); return tmp;}

    sysinline Self operator --(int)
        {Self tmp(*this); --(*this); return tmp;}

    //
    // +=
    // -=
    //

public:

    sysinline Self& operator +=(Space diff)
    {
        base.currentPtr += (DbgptrAddrU(diff) * sizeof(Element));
        return *this;
    }

    sysinline Self& operator -=(Space diff)
    {
        base.currentPtr -= (DbgptrAddrU(diff) * sizeof(Element));
        return *this;
    }

    //
    // addByteOffset
    //

    sysinline void addByteOffset(DbgptrAddrS ofs)
    {
        base.currentPtr += ofs;
    }

    //
    // * Get reference
    //

public:

    sysinline const DbgptrReference<Self>& operator *() const
        {return (const DbgptrReference<Self>&) *this;}

    //
    // Operator []
    //

public:

    sysinline auto& operator [](Space index) const
    {
        DbgptrReference<Self> result = (const DbgptrReference<Self>&) *this;
        result.asPointer() += index;
        return result.modify();
    }

    //
    // Operator ->
    //

public:

    sysinline Element* operator ->() const
        {return &modify();}

    //
    // Operator +(n)
    // Operator -(n)
    //

public:

    sysinline Self operator +(Space index) const
    {
        Self tmp = *this; tmp += index;
        return tmp;
    }

    sysinline Self operator -(Space index) const
    {
        Self tmp = *this; tmp -= index;
        return tmp;
    }

    //
    // Operator -(p)
    //

public:

    sysinline Space operator -(const Self& that) const
        {return Space(((Element*) this->base.currentPtr) - ((Element*) that.base.currentPtr));}

    //
    // Comparisons
    //

public:

    #define TMP_MACRO(OP) \
        sysinline bool operator OP(const Self& that) const \
            {return this->base.currentPtr OP that.base.currentPtr;}

    TMP_MACRO(>)
    TMP_MACRO(<)
    TMP_MACRO(>=)
    TMP_MACRO(<=)
    TMP_MACRO(==)
    TMP_MACRO(!=)

    #undef TMP_MACRO

    //
    // Validate range
    //

public:

    void validateRange(Space size) const
    {
        if_not (size != 0) return;
        bool ok = (SpaceU(size) <= SpaceU(maxSize));
        base.validateArray(ok, size * sizeof(Element));
    }

public:

    const DebugArrayPointerByteEngine& getBaseRef() const
        {return base;}

private:

    DebugArrayPointerByteEngine base;

};

//----------------------------------------------------------------

COMPILE_ASSERT_EQUAL_LAYOUT(DbgptrReference<DebugArrayPointer<int>>, DebugArrayPointer<int>);

//================================================================
//
// unsafePtr<DebugArrayPointer>
//
//================================================================

template <typename Element>
sysinline Element* unsafePtr(const DebugArrayPointer<Element>& ptr, Space size)
{
    ptr.validateRange(size);
    return ptr.getPtrForInternalUsageOnly();
}

//================================================================
//
// isPtrAligned
//
//================================================================

template <Space alignment, typename Type>
sysinline bool isPtrAligned(const DebugArrayPointer<Type>& ptr)
{
    return isPtrAligned<alignment>(ptr.getPtrForInternalUsageOnly());
}

//================================================================
//
// PtrElemType
//
//================================================================

template <typename Type>
struct PtrElemType<DebugArrayPointer<Type>>
{
    using T = Type;
};

//================================================================
//
// addOffset
//
//================================================================

template <typename Type>
sysinline DebugArrayPointer<Type> addOffset(const DebugArrayPointer<Type>& ptr, DbgptrAddrS ofs)
{
    auto result = ptr;
    result.addByteOffset(ofs);
    return result;
}
