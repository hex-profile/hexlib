#pragma once

#include "dbgptr/dbgptrArrayPointer.h"
#include "data/pointerInterface.h"

//================================================================
//
// DebugMatrixPointerByteEngine
//
//================================================================

class DebugMatrixPointerByteEngine
{

    //
    // Default constructor
    //

public:

    struct ConstructUnintialized {};
    explicit DebugMatrixPointerByteEngine(const ConstructUnintialized&) {}
    void initEmpty();

    //
    // Construct by matrix.
    //
    // PRECONDITIONS:
    //
    // (1) sizeX >= 0 && sizeY >= 0
    // (2) sizeX <= |pitch|
    // (3) (sizeX * pitch) fits into Space type
    //

public:

    struct PreconditionsAreValid {};

    void initByMatrix(DbgptrAddrU matrMemPtr, Space matrMemPitch, Space matrSizeX, Space matrSizeY, const PreconditionsAreValid&);

    //
    // Copy
    //

public:

    DebugMatrixPointerByteEngine(const DebugMatrixPointerByteEngine& that);
    DebugMatrixPointerByteEngine& operator =(const DebugMatrixPointerByteEngine& that);

private:

    sysinline void copyFrom(const DebugMatrixPointerByteEngine& that);

    //
    // Import array pointer
    //

public:

    void initByArrayPointer(const DebugArrayPointerByteEngine& ptr);

    //
    // Export array pointer
    //

public:

    void exportArrayPointer(DebugArrayPointerByteEngine& result) const;

    //
    // Validate a single byte. The tested size == 1. The fastest test.
    //

public:

    sysinline void validateSingleByte() const
    {
        if_not ((currentPtr - cachedRow) < matrixSizeX)
            validateSingleByteSlow();
    }

private:

    void validateSingleByteSlow() const;

    //
    // Validate 1D byte range inside single matrix row.
    //

public:

    sysinline void validateArray(bool ok, DbgptrAddrU testSizeX) const
    {
        bool fastOk = ok;
        check_flag(testSizeX <= matrixSizeX, fastOk);
        check_flag(currentPtr - cachedRow <= matrixSizeX - testSizeX, fastOk);

        if_not (fastOk)
            validateArraySlow(ok, testSizeX);
    }

private:

    void validateArraySlow(bool ok, DbgptrAddrU testSizeX) const;

    //
    // Validate 2D byte range.
    //

public:

    void validateMatrix(bool ok, DbgptrAddrU testSizeX, DbgptrAddrU testSizeY) const;

    //
    // Row update
    //

private:

    bool rowUpdateSlow() const;
    sysinline bool rowUpdate() const;

public:

    // Current byte pointer.
    // (First class member for better performance)
    DbgptrAddrU currentPtr;

private:

    // Byte pointer to the matrix memory beginning.
    DbgptrAddrU matrixStart;

    // Byte distance between rows. |matrixPitch| >= matrixSizeX.
    // The pitch can be negative. Negative sign indicates inverse row order,
    // without any other impact: code always uses |matrixPitch|.
    Space matrixPitch;

    // The byte size of matrix row, range [0, |matrixPitch|].
    // The number of matrix rows, >= 0.
    DbgptrAddrU matrixSizeX;
    DbgptrAddrU matrixSizeY;

    // Pointer to the the cached row beginning.
    // It doesn't have to point to the current pointer's row,
    // but it SHOULD point to some row inside matrix.
    mutable DbgptrAddrU cachedRow;

};

//================================================================
//
// DbgptrMatrixPreconditions:
//
// (1) sizeX >= 0 && sizeY >= 0
// (2) sizeX <= |pitch|
// (3) (sizeX * pitch * sizeof(Element)) fits into Space type
//
//================================================================

struct DbgptrMatrixPreconditions {};

//================================================================
//
// DebugMatrixPointer
//
// The class is based on DebugMatrixPointerByteEngine.
//
// When checking for a single element, validateSingleByte is used.
//
// This is identical to checking the whole element, because
// * row size is multiple of element size E AND
// * (start of any row mod E) == (current pointer mod E).
//
// The latest assumption is valid because:
// * pitch is multiple of E AND
// * (base address mod E) == (current pointer mod E).
//
// This requires ALL modifications of the current pointer to be multiples of element size!
//
//================================================================

template <typename Type>
class DebugMatrixPointer
{

    template <typename Any>
    friend class DebugMatrixPointer;

public:

    using Element = Type;

    using Self = DebugMatrixPointer<Element>;

    //
    // Create empty
    //

public:

    sysinline DebugMatrixPointer()
        :
        base(DebugMatrixPointerByteEngine::ConstructUnintialized())
    {
        base.initEmpty();
    }

public:

    struct Null;

    sysinline DebugMatrixPointer(Null*)
        :
        base(DebugMatrixPointerByteEngine::ConstructUnintialized())
    {
        base.initEmpty();
    }

    //
    // Create by matrix
    //

public:

    sysinline DebugMatrixPointer(Element* matrMemPtr, Space matrMemPitch, Space matrSizeX, Space matrSizeY, const DbgptrMatrixPreconditions&)
        :
        base(DebugMatrixPointerByteEngine::ConstructUnintialized())
    {
        base.initByMatrix
        (
            DbgptrAddrU(matrMemPtr),
            matrMemPitch * Space(sizeof(Element)),
            matrSizeX * Space(sizeof(Element)),
            matrSizeY,
            DebugMatrixPointerByteEngine::PreconditionsAreValid()
        );
    }

public:

    sysinline DebugMatrixPointer(const DebugArrayPointer<Element>& that)
        :
        base(DebugMatrixPointerByteEngine::ConstructUnintialized())
    {
        base.initByArrayPointer(that.getBaseRef());
    }

    //
    // Copy
    //

public:

    sysinline DebugMatrixPointer(const Self& that)
        :
        base(that.base)
    {
    }

    sysinline DebugMatrixPointer<Element>& operator =(const Self& that)
    {
        base = that.base;
        return *this;
    }

    //
    // Advanced copy
    //

public:

    template <typename Other>
    sysinline DebugMatrixPointer(const DebugMatrixPointer<Other>& that)
        :
        base(that.base)
    {
        Element* checkConversion = (Other*) 0; checkConversion = checkConversion;
    }

    template <typename Other>
    sysinline DebugMatrixPointer<Element>& operator =(const DebugMatrixPointer<Other>& that)
    {
        Element* checkConversion = (Other*) 0; checkConversion = checkConversion;

        base = that.base;
        return *this;
    }

    //
    // Export array pointer
    //

public:

    sysinline operator DebugArrayPointer<Element> ()
    {
        DebugArrayPointerByteEngine result;
        base.exportArrayPointer(result);
        return DebugArrayPointer<Element>(result);
    }

    //
    // Internal pointer
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
    {
        base.currentPtr += sizeof(Element);
        return *this;
    }

    sysinline Self& operator --()
    {
        base.currentPtr -= sizeof(Element);
        return *this;
    }

    sysinline Self operator ++(int)
        {Self tmp = *this; ++(*this); return tmp;}

    sysinline Self operator --(int)
        {Self tmp = *this; --(*this); return tmp;}

    //
    // +=
    // -=
    //

public:

    sysinline Self& operator +=(Space diff)
    {
        base.currentPtr += DbgptrAddrU(diff) * sizeof(Element);
        return *this;
    }

    sysinline Self& operator -=(Space diff)
    {
        base.currentPtr -= DbgptrAddrU(diff) * sizeof(Element);
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
    // Basic access
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
    // Dereference
    // operator []
    // operator ->
    //

public:

    sysinline const DbgptrReference<Self>& operator *() const
        {return (const DbgptrReference<Self>&) *this;}

    sysinline DbgptrReference<Self> operator [](Space index) const
    {
        DbgptrReference<Self> result = (const DbgptrReference<Self>&) *this;
        result.asPointer() += index;
        return result;
    }

    sysinline Element* operator ->() const
        {return &modify();}

    //
    // +
    // -
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
    // Difference of pointers
    //

public:

    sysinline Space operator -(const Self& that)
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
    // Validate 1D range
    //

private:

    static const Space maxSize = spaceMax / sizeof(Element);

public:

    void validateRange1D(Space testSizeX) const
    {
        if_not (testSizeX != 0) return;
        bool ok = (SpaceU(testSizeX) <= SpaceU(maxSize));
        base.validateArray(ok, testSizeX * Space(sizeof(Element)));
    }

    //
    // Validate 2D range.
    //

public:

    sysinline void validateRange2D(Space testSizeX, Space testSizeY) const
    {
        if_not (testSizeX != 0 && testSizeY != 0) return;

        bool ok = SpaceU(testSizeX) <= SpaceU(maxSize);
        base.validateMatrix(ok, testSizeX * sizeof(Element), testSizeY);
    }

private:

    DebugMatrixPointerByteEngine base;

};

//----------------------------------------------------------------

COMPILE_ASSERT_EQUAL_LAYOUT(DbgptrReference<DebugMatrixPointer<int>>, DebugMatrixPointer<int>);

//================================================================
//
// unsafePtr
//
//================================================================

template <typename Element>
sysinline Element* unsafePtr(const DebugMatrixPointer<Element>& ptr, Space sizeX)
{
    ptr.validateRange1D(sizeX);
    return ptr.getPtrForInternalUsageOnly();
}

//----------------------------------------------------------------

template <typename Element>
sysinline Element* unsafePtr(const DebugMatrixPointer<Element>& ptr, Space sizeX, Space sizeY)
{
    ptr.validateRange2D(sizeX, sizeY);
    return ptr.getPtrForInternalUsageOnly();
}

//================================================================
//
// isPtrAligned
//
//================================================================

template <Space alignment, typename Type>
sysinline bool isPtrAligned(const DebugMatrixPointer<Type>& ptr)
{
    return isPtrAligned<alignment>(ptr.getPtrForInternalUsageOnly());
}

//================================================================
//
// PtrElemType
//
//================================================================

template <typename Type>
struct PtrElemType<DebugMatrixPointer<Type>>
{
    using T = Type;
};

//================================================================
//
// addOffset
//
//================================================================

template <typename Type>
sysinline DebugMatrixPointer<Type> addOffset(const DebugMatrixPointer<Type>& ptr, DbgptrAddrS ofs)
{
    auto result = ptr;
    result.addByteOffset(ofs);
    return result;
}
