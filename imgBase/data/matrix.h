#pragma once

#include "data/array.h"
#include "data/commonFuncs.h"
#include "data/pointerInterface.h"
#include "data/spacex.h"
#include "errorLog/debugBreak.h"
#include "extLib/data/matrixBase.h"
#include "numbers/int/intBase.h"
#include "point/point.h"
#include "storage/addrSpace.h"

//================================================================
//
// Matrix<Type>
//
// Matrix memory layout: base pointer, pitch and dimensions.
//
//----------------------------------------------------------------
//
// memPtr:
// Points to (0, 0) element. Can be undefined if the matrix is empty.
//
// memPitch:
// The difference of pointers to (X, Y+1) and (X, Y) elements.
// The difference is expressed in elements (not bytes). Can be negative.
//
// sizeX, sizeY:
// The width and height of the matrix. Both are >= 0.
// If either of them is zero, the matrix is empty.
//
//================================================================

//================================================================
//
// MATRIX__CHECK_POINTER
//
//================================================================

template <typename SrcPointer, typename DstPointer>
inline auto matrixCheckPointerConversion()
{
    SrcPointer srcPtr(0);
    DstPointer dstPtr = srcPtr;
    return dstPtr;
}

//----------------------------------------------------------------

#define MATRIX__CHECK_POINTER(SrcPointer, DstPointer) \
    COMPILE_ASSERT(sizeof(matrixCheckPointerConversion<SrcPointer, DstPointer>()) >= 0)

//================================================================
//
// MatrixProps
//
//================================================================

template <typename Matrix>
struct MatrixProps
{

private:

    template <typename Type, typename Pointer, typename Pitch>
    static Pitch getPitchType(const MatrixBase<Type, Pointer, Pitch>& matrix);

    ////

    static const Matrix matrixExample;

public:

    using Pitch = decltype(getPitchType(matrixExample));

    ////

    COMPILE_ASSERT(TYPE_EQUAL(Pitch, PitchMayBeNegative) || TYPE_EQUAL(Pitch, PitchPositiveOrZero));
    static constexpr bool pitchIsNonNeg = TYPE_EQUAL(Pitch, PitchPositiveOrZero);

};

//================================================================
//
// MATRIX_EXPOSE
//
//================================================================

#define MATRIX_EXPOSE_GENERIC(matrix, prefix) \
    auto prefix##MemPtr = (matrix).memPtr(); \
    auto prefix##MemPitch = (matrix).memPitch(); \
    auto prefix##SizeX = (matrix).sizeX(); \
    auto prefix##SizeY = (matrix).sizeY(); \
    constexpr bool prefix##PitchIsNonNeg = MatrixProps<decltype(matrix)>::pitchIsNonNeg; \
    (void) prefix##PitchIsNonNeg

//----------------------------------------------------------------

#define MATRIX_EXPOSE(matrix) \
    MATRIX_EXPOSE_GENERIC(matrix, matrix)

#define MATRIX_EXPOSE_EX(matrix, prefix) \
    MATRIX_EXPOSE_GENERIC(matrix, prefix)

//----------------------------------------------------------------

#define MATRIX_EXPOSE_UNSAFE_EX(matrix, prefix) \
    auto prefix##MemPtr = (matrix).memPtrUnsafeInternalUseOnly(); \
    auto prefix##MemPitch = (matrix).memPitch(); \
    auto prefix##SizeX = (matrix).sizeX(); \
    auto prefix##SizeY = (matrix).sizeY(); \
    constexpr bool prefix##PitchIsNonNeg = MatrixProps<decltype(matrix)>::pitchIsNonNeg; \
    (void) prefix##PitchIsNonNeg

#define MATRIX_EXPOSE_UNSAFE(matrix) \
    MATRIX_EXPOSE_UNSAFE_EX(matrix, matrix)

//================================================================
//
// MATRIX_VALID_ACCESS
//
//================================================================

#define MATRIX_VALID_ACCESS(matrix, X, Y) \
    \
    ( \
        SpaceU(X) < SpaceU(matrix##SizeX) && \
        SpaceU(Y) < SpaceU(matrix##SizeY) \
    )

#define MATRIX_VALID_ACCESS_(matrix, pos) \
    MATRIX_VALID_ACCESS(matrix, (pos).X, (pos).Y)

////

sysinline bool matrixValidAccess(const Point<Space>& size, const Point<Space>& pos)
{
    return
        SpaceU(pos.X) < SpaceU(size.X) &&
        SpaceU(pos.Y) < SpaceU(size.Y);
}

//================================================================
//
// MatrixPointerComputation
//
//================================================================

template <bool pitchIsNonNeg>
struct MatrixPointerComputation;

////

template <>
struct MatrixPointerComputation<false>
{
    template <typename Pointer>
    static sysinline Pointer get(Pointer memPtr, Space memPitch, Space X, Space Y)
    {
        using Type = typename PtrElemType<Pointer>::T;

        //
        // In this mode, the pitch can be negative, which generates a multitude
        // of redundant commands in 64-bit mode, including on the GPU.
        //
        // Although (X + Y * pitch) is calculated in a 32-bit type, in subsequent
        // operations, if you simply add the element index to the pointer,
        // multiplication by the type size is performed in a 64-bit signed type,
        // which is not as efficient.
        //
        // All that can be done here is at least to calculate the offset in bytes
        // in a 32-bit signed type, which will then, nevertheless, be signedly
        // added to a 64-bit pointer.
        //

        Space offsetInBytes = (X + Y * memPitch) * Space(sizeof(Type));

        return addOffset(memPtr, offsetInBytes);
    }
};

////

template <>
struct MatrixPointerComputation<true>
{
    template <typename Pointer>
    static sysinline Pointer get(Pointer memPtr, Space memPitch, Space X, Space Y)
    {
        using Type = typename PtrElemType<Pointer>::T;

        //
        // In this mode, pitch >= 0, while the coordinates X and Y can only
        // be negative in the event of a matrix boundary violation.
        //
        // Therefore, all calculations up to the addition of the byte offset
        // to the pointer can be performed in a 32-bit unsigned type,
        // providing the most efficient code in 64-bit mode, including on the GPU.
        //

        SpaceU offsetInBytes = (SpaceU(X) + SpaceU(Y) * SpaceU(memPitch)) * SpaceU(sizeof(Type));

        return addOffset(memPtr, offsetInBytes);
    }
};

//================================================================
//
// MATRIX_POINTER
// MATRIX_ELEMENT
//
//================================================================

#define MATRIX_POINTER(matrix, X, Y) \
    (MatrixPointerComputation<matrix##PitchIsNonNeg>::get(matrix##MemPtr, matrix##MemPitch, X, Y))

#define MATRIX_POINTER_(matrix, pos) \
    MATRIX_POINTER(matrix, (pos).X, (pos).Y)

//----------------------------------------------------------------

#define MATRIX_ELEMENT(matrix, X, Y) \
    (*MATRIX_POINTER(matrix, X, Y))

#define MATRIX_ELEMENT_(matrix, pos) \
    MATRIX_ELEMENT(matrix, (pos).X, (pos).Y)

//----------------------------------------------------------------

#define MATRIX_READ(matrix, X, Y) \
    helpRead(MATRIX_ELEMENT(matrix, X, Y))

//================================================================
//
// matrixParamsAreValid
//
//================================================================

template <Space elemSize, typename Pitch>
bool matrixParamsAreValid(Space sizeX, Space sizeY, Space pitch);

//================================================================
//
// MatrixCreateFromArray
//
//================================================================

struct MatrixCreateFromArray {};

//================================================================
//
// MatrixEx<Type>
//
// Generic Matrix class, supports any pointer type.
//
//================================================================

template <typename Pointer, typename Pitch = PitchDefault>
class MatrixEx
    :
    public MatrixBase<typename PtrElemType<Pointer>::T, Pointer, Pitch>
{

    //----------------------------------------------------------------
    //
    // Types.
    //
    //----------------------------------------------------------------

public:

    using Type = typename PtrElemType<Pointer>::T;

private:

    template <typename, typename>
    friend class MatrixEx;

    using Base = MatrixBase<Type, Pointer, Pitch>;
    using Self = MatrixEx<Pointer, Pitch>;

    using Base::theMemPtrUnsafe;
    using Base::theMemPitch;
    using Base::theSizeX;
    using Base::theSizeY;

public:

    //----------------------------------------------------------------
    //
    // Constructors.
    //
    //----------------------------------------------------------------

    sysinline MatrixEx()
        {}

    ////

    struct TmpType {};

    sysinline MatrixEx(const TmpType*)
        {}

    //----------------------------------------------------------------
    //
    // Create by an array.
    //
    //----------------------------------------------------------------

    template <typename SrcPointer>
    sysinline MatrixEx(const ArrayEx<SrcPointer>& that, const MatrixCreateFromArray&)
    {
        MATRIX__CHECK_POINTER(SrcPointer, Pointer);
        assignUnsafe(that.ptr(), that.size(), that.size(), 1);
    }

    //----------------------------------------------------------------
    //
    // Export cast (no code generated, reinterpret pointer).
    //
    // Initial design involved template-based type-casting operator. However,
    // this led to compatibility issues with GCC.
    //
    // To work around this, explicit type-casting operators are defined for all
    // specific scenarios. These operators conditionally return a const reference
    // to either a modified `MatrixEx` type or a placeholder `DummyType`.
    //
    // Flags like `exportPointerAvail` and `exportPitchAvail` are used to
    // conditionally enable these type conversions.
    //
    //----------------------------------------------------------------

    using ExportPointer = typename PtrRebaseType<Pointer, const Type>::T;
    static constexpr bool exportPointerAvail = !TYPE_EQUAL(Pointer, ExportPointer);

    using ExportPitch = PitchMayBeNegative;
    static constexpr bool exportPitchAvail = !TYPE_EQUAL(Pitch, ExportPitch);

    ////

    struct DummyType1;
    using ExportType1 = TypeSelect<exportPointerAvail, MatrixEx<ExportPointer, Pitch>, DummyType1>;

    struct DummyType2;
    using ExportType2 = TypeSelect<exportPitchAvail, MatrixEx<Pointer, ExportPitch>, DummyType2>;

    struct DummyType3;
    using ExportType3 = TypeSelect<exportPointerAvail && exportPitchAvail, MatrixEx<ExportPointer, ExportPitch>, DummyType3>;

    ////

    sysinline operator const ExportType1& () const
        {return recastEqualLayout<const ExportType1>(*this);}

    sysinline operator const ExportType2& () const
        {return recastEqualLayout<const ExportType2>(*this);}

    sysinline operator const ExportType3& () const
        {return recastEqualLayout<const ExportType3>(*this);}

    //----------------------------------------------------------------
    //
    // assignValidated
    //
    //----------------------------------------------------------------

    sysnodiscard
    sysinline bool assignValidated(Pointer memPtr, Space memPitch, Space sizeX, Space sizeY)
    {
        bool ok = matrixParamsAreValid<sizeof(Type), Pitch>(sizeX, sizeY, memPitch);
        ensure(DEBUG_BREAK_CHECK(ok));

        theMemPtrUnsafe = memPtr;
        theMemPitch = memPitch;
        theSizeX = sizeX;
        theSizeY = sizeY;
        return true;
    }

    //----------------------------------------------------------------
    //
    // assignUnsafe
    //
    //----------------------------------------------------------------

    sysinline void assignUnsafe(Pointer memPtr, Space memPitch, Space sizeX, Space sizeY)
    {
        constexpr Space maxArea = spaceMax / Space(sizeof(Type));

        if_not // quick check
        (
            SpaceU(sizeX) <= SpaceU(maxArea) &&
            SpaceU(sizeY) <= SpaceU(maxArea) &&
            (TYPE_EQUAL(Pitch, PitchMayBeNegative) || memPitch >= 0)
        )
        {
            DEBUG_BREAK_INLINE();
            sizeX = 0;
            sizeY = 0;
        }

        theMemPtrUnsafe = memPtr;
        theMemPitch = memPitch;
        theSizeX = sizeX;
        theSizeY = sizeY;
    }

    //----------------------------------------------------------------
    //
    // assign*<MatrixPtr>
    //
    //----------------------------------------------------------------

#if HEXLIB_GUARDED_MEMORY

    bool assignValidated(MatrixPtr(Type) memPtr, Space memPitch, Space sizeX, Space sizeY)
    {
        return assignValidated(unsafePtr(memPtr, sizeX, sizeY), memPitch, sizeX, sizeY);
    }

    void assignUnsafe(MatrixPtr(Type) memPtr, Space memPitch, Space sizeX, Space sizeY)
    {
        if_not (assignValidated(memPtr, memPitch, sizeX, sizeY))
            {DEBUG_BREAK_INLINE(); assignNull();}
    }

#endif

    //----------------------------------------------------------------
    //
    // assign*<ArrayPtr>
    //
    //----------------------------------------------------------------

#if HEXLIB_GUARDED_MEMORY

    sysnodiscard
    sysinline bool assignValidated(ArrayPtr(Type) memPtr, Space memPitch, Space sizeX, Space sizeY)
    {
        assignNull();
        ensure(memPitch >= 0);

        ////

        Space area = 0;
        ensure(safeMul(memPitch, sizeY, area));

        ////

        return assignValidated(unsafePtr(memPtr, area), memPitch, sizeX, sizeY);
    }

    sysinline void assignUnsafe(ArrayPtr(Type) memPtr, Space memPitch, Space sizeX, Space sizeY)
    {
        if_not (assignValidated(memPtr, memPitch, sizeX, sizeY))
        {
            DEBUG_BREAK_INLINE();
            assignNull();
        }
    }

#endif

    //----------------------------------------------------------------
    //
    // Assign empty
    //
    //----------------------------------------------------------------

    sysinline void assignNull()
    {
        theMemPtrUnsafe = Pointer(0);
        theMemPitch = 0;
        theSizeX = 0;
        theSizeY = 0;
    }

    //----------------------------------------------------------------
    //
    // Get size. Always >= 0.
    //
    //----------------------------------------------------------------

    sysinline Space sizeX() const
        {return theSizeX;}

    sysinline Space sizeY() const
        {return theSizeY;}

    sysinline Point<Space> size() const
        {return {theSizeX, theSizeY};}

    //----------------------------------------------------------------
    //
    // Get pitch.
    //
    //----------------------------------------------------------------

    sysinline Space memPitch() const
        {return theMemPitch;}

    //----------------------------------------------------------------
    //
    // Get base pointer.
    //
    //----------------------------------------------------------------

    sysinline Pointer memPtrUnsafeInternalUseOnly() const
        {return theMemPtrUnsafe;}

#if HEXLIB_GUARDED_MEMORY

    sysinline MatrixPtr(Type) memPtr() const
        {return MatrixPtrCreate(Type, theMemPtrUnsafe, theMemPitch, theSizeX, theSizeY, DbgptrMatrixPreconditions());}

#else

    sysinline Pointer memPtr() const
        {return theMemPtrUnsafe;}

#endif

    //----------------------------------------------------------------
    //
    // subr
    //
    // Cuts a rectangular area from matrix, given by origin and end: point RIGHT AFTER the last element.
    // If the area is larger than the matrix, it is clipped to fit the matrix and false is returned.
    //
    //----------------------------------------------------------------

    template <typename DestPointer, typename DestPitch>
    sysinline bool subr(Space orgX, Space orgY, Space endX, Space endY, MatrixEx<DestPointer, DestPitch>& result) const
    {
        MATRIX__CHECK_POINTER(Pointer, DestPointer);
        PitchCheckConversion<Pitch, DestPitch>{};

        MATRIX_EXPOSE_UNSAFE_EX(*this, my);

        Space clOrgX = clampRange(orgX, 0, mySizeX);
        Space clOrgY = clampRange(orgY, 0, mySizeY);

        Space clEndX = clampRange(endX, clOrgX, mySizeX);
        Space clEndY = clampRange(endY, clOrgY, mySizeY);

        result.theMemPtrUnsafe = MATRIX_POINTER(my, clOrgX, clOrgY);
        result.theMemPitch = myMemPitch;

        result.theSizeX = clEndX - clOrgX;
        result.theSizeY = clEndY - clOrgY;

        return
            (clOrgX == orgX) &&
            (clOrgY == orgY) &&
            (clEndX == endX) &&
            (clEndY == endY);
    }

    ////

    template <typename DestPointer, typename DestPitch>
    sysinline bool subr(const Point<Space>& org, const Point<Space>& end, MatrixEx<DestPointer, DestPitch>& result) const
    {
        return subr(org.X, org.Y, end.X, end.Y, result);
    }

    //----------------------------------------------------------------
    //
    // subs
    //
    // Cuts a rectangular area from matrix, given by starting point and size.
    //
    // If the area is larger than the matrix, it is clipped to fit the matrix
    // and false is returned.
    //
    //----------------------------------------------------------------

    template <typename DestPointer, typename DestPitch>
    sysinline bool subs(Space orgX, Space orgY, Space sizeX, Space sizeY, MatrixEx<DestPointer, DestPitch>& result) const
    {
        MATRIX__CHECK_POINTER(Pointer, DestPointer);
        PitchCheckConversion<Pitch, DestPitch>{};

        MATRIX_EXPOSE_UNSAFE_EX(*this, my);

        Space clOrgX = clampRange(orgX, 0, mySizeX);
        Space clOrgY = clampRange(orgY, 0, mySizeY);

        Space clSizeX = clampRange(sizeX, 0, mySizeX - clOrgX);
        Space clSizeY = clampRange(sizeY, 0, mySizeY - clOrgY);

        result.theMemPtrUnsafe = MATRIX_POINTER(my, clOrgX, clOrgY);
        result.theMemPitch = myMemPitch;

        result.theSizeX = clSizeX;
        result.theSizeY = clSizeY;

        return
            (clOrgX == orgX) &&
            (clOrgY == orgY) &&
            (clSizeX == sizeX) &&
            (clSizeY == sizeY);
    }

    ////

    template <typename DestPointer, typename DestPitch>
    sysinline bool subs(const Point<Space>& org, const Point<Space>& size, MatrixEx<DestPointer, DestPitch>& result) const
    {
        return subs(org.X, org.Y, size.X, size.Y, result);
    }

    //----------------------------------------------------------------
    //
    // asArray
    //
    // Tries to export the matrix as array.
    // This is possible only for dense row allocation (pitch == size.X)
    //
    //----------------------------------------------------------------

    template <typename DestPointer>
    sysinline bool asArray(ArrayEx<DestPointer>& result) const
    {
        MATRIX__CHECK_POINTER(Pointer, DestPointer);

        bool ok = true;

        check_flag(theMemPitch == theSizeX, ok);

        Space totalSize = theSizeX * theSizeY;

        if_not (ok)
            totalSize = 0;

        result.assignUnsafe(theMemPtrUnsafe, totalSize);

        return ok;
    }

    //----------------------------------------------------------------
    //
    // validAccess
    //
    //----------------------------------------------------------------

    sysinline bool validAccess(const Point<Space>& pos) const
    {
        MATRIX_EXPOSE_EX(*this, my);
        return MATRIX_VALID_ACCESS_(my, pos);
    }

    sysinline bool validAccess(Space X, Space Y) const
    {
        MATRIX_EXPOSE_EX(*this, my);
        return MATRIX_VALID_ACCESS(my, X, Y);
    }

    //----------------------------------------------------------------
    //
    // Pointer, reference, read: direct access, checked only in guarded mode.
    //
    //----------------------------------------------------------------

    sysinline auto pointer(const Point<Space>& pos) const
    {
        MATRIX_EXPOSE_EX(*this, my);
        return MATRIX_POINTER_(my, pos);
    }

    sysinline auto pointer(Space X, Space Y) const
    {
        MATRIX_EXPOSE_EX(*this, my);
        return MATRIX_POINTER(my, X, Y);
    }

    ////

    sysinline auto& element(const Point<Space>& pos) const
    {
        MATRIX_EXPOSE_EX(*this, my);
        return helpModify(MATRIX_ELEMENT_(my, pos));
    }

    sysinline auto& element(Space X, Space Y) const
    {
        MATRIX_EXPOSE_EX(*this, my);
        return helpModify(MATRIX_ELEMENT(my, X, Y));
    }

    ////

    sysinline auto read(const Point<Space>& pos) const
    {
        MATRIX_EXPOSE_EX(*this, my);
        return helpRead(MATRIX_ELEMENT_(my, pos));
    }

    sysinline auto read(Space X, Space Y) const
    {
        MATRIX_EXPOSE_EX(*this, my);
        return helpRead(MATRIX_ELEMENT(my, X, Y));
    }

    //----------------------------------------------------------------
    //
    // writeSafe
    //
    //----------------------------------------------------------------

    template <typename Value>
    sysinline void writeSafe(const Point<Space>& pos, const Value& value) const
    {
        MATRIX_EXPOSE_UNSAFE_EX(*this, my);

        if (MATRIX_VALID_ACCESS_(my, pos))
            helpModify(MATRIX_ELEMENT_(my, pos)) = helpRead(value);
    }

    template <typename Value>
    sysinline void writeSafe(Space X, Space Y, const Value& value) const
    {
        MATRIX_EXPOSE_UNSAFE_EX(*this, my);

        if (MATRIX_VALID_ACCESS(my, X, Y))
            helpModify(MATRIX_ELEMENT(my, X, Y)) = helpRead(value);
    }

    //----------------------------------------------------------------
    //
    // exchange
    //
    //----------------------------------------------------------------

    sysinline friend void exchange(Self& A, Self& B)
    {
        exchange(A.theMemPtrUnsafe, B.theMemPtrUnsafe);
        exchange(A.theMemPitch, B.theMemPitch);
        exchange(A.theSizeX, B.theSizeX);
        exchange(A.theSizeY, B.theSizeY);
    }

};

//----------------------------------------------------------------

COMPILE_ASSERT_EQUAL_LAYOUT(MatrixEx<int*>, MatrixBase<int>);

//================================================================
//
// makeConst (fast)
//
//================================================================

template <typename Type, typename Pitch>
sysinline auto& makeConst(const MatrixEx<Type*, Pitch>& matrix)
{
    return recastEqualLayout<const MatrixEx<const Type*, Pitch>>(matrix);
}

//================================================================
//
// relaxToAnyPitch
//
//================================================================

template <typename Pointer, typename Pitch>
sysinline auto& relaxToAnyPitch(const MatrixEx<Pointer, Pitch>& matrix)
    {return recastEqualLayout<const MatrixEx<Pointer, PitchMayBeNegative>>(matrix);}

//================================================================
//
// restrictToNonNegativePitch
//
// Use with caution!
//
//================================================================

template <typename Pointer, typename Pitch>
sysinline auto& restrictToNonNegativePitch(const MatrixEx<Pointer, Pitch>& matrix)
    {return recastEqualLayout<const MatrixEx<Pointer, PitchPositiveOrZero>>(matrix);}

//================================================================
//
// equalSize support
//
//================================================================

template <>
GET_SIZE_DEFINE(Point<Space>, value)

template <typename Pointer, typename Pitch>
GET_SIZE_DEFINE(MatrixEx<Pointer PREP_COMMA Pitch>, value.size())

template <typename Pointer, typename Pitch>
sysinline Space getLayers(const MatrixEx<Pointer, Pitch>& matrix)
    {return 1;}

//================================================================
//
// hasData
//
//================================================================

template <typename Type, typename Pitch>
sysinline bool hasData(const MatrixEx<Type, Pitch>& matrix)
    {return allv(matrix.size() >= 1);}

sysinline bool hasData(const Point<Space>& size)
    {return allv(size >= 1);}

//----------------------------------------------------------------

template <typename Type>
sysinline bool empty(const Type& V)
    {return !hasData(V);}

//================================================================
//
// areaOf
//
//================================================================

template <typename Type>
sysinline Space areaOf(const Type& value)
{
    Point<Space> size = GetSize<Type>::func(value);
    return size.X * size.Y;
}

//================================================================
//
// matrix
//
//================================================================

template <typename Pointer>
sysinline auto matrix(const ArrayEx<Pointer>& arr)
{
    return MatrixEx<Pointer, PitchPositiveOrZero>(arr, MatrixCreateFromArray{});
}

//================================================================
//
// Matrix
//
// Matrix for C++ address space: identical to MatrixEx<Type*>.
//
//================================================================

template <typename Type, typename Pitch = PitchDefault>
using Matrix = MatrixEx<Type*, Pitch>;

//================================================================
//
// MatrixAP
//
//================================================================

template <typename Type>
using MatrixAP = MatrixEx<Type*, PitchMayBeNegative>;

//================================================================
//
// recastElement
//
// Use with caution!
//
//================================================================

template <typename Dst, typename Src, typename Pitch>
sysinline auto& recastElement(const MatrixEx<Src*, Pitch>& matrix)
{
    COMPILE_ASSERT_EQUAL_LAYOUT(Src, Dst);
    return recastEqualLayout<const MatrixEx<Dst*, Pitch>>(matrix);
}
