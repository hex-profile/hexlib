#pragma once

#include "data/space.h"
#include "point/point.h"
#include "data/array.h"
#include "data/pointerInterface.h"
#include "numbers/int/intType.h"

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
// USAGE EXAMPLES:
//
//================================================================

#if 0

// Create an empty matrix.
Matrix<int> intMatrix;
Matrix<int> anotherEmptyMatrix = 0;

// Convert a matrix to a read-only matrix.
Matrix<const int> constIntMatrix = intMatrix;
Matrix<const int> anotherConstMatrix = makeConst(intMatrix);

// Construct matrix from details: ptr, pitch and size.
Matrix<const uint8> example(srcMemPtrUnsafe, srcMemPitch, srcSizeX, srcSizeY);

// Setup matrix from details: ptr, pitch and size.
example.assign(srcMemPtrUnsafe, srcMemPitch, srcSizeX, srcSizeY);

// Make the matrix empty.
example.assignNull();

// Access matrix details (decomposing matrix is better way):
REQUIRE(example.memPtr() != 0);
REQUIRE(example.memPitch() != 0);
REQUIRE(example.sizeX() != 0);
REQUIRE(example.sizeY() != 0);

// Decompose a matrix to detail variables:
MATRIX_EXPOSE(example);
REQUIRE(exampleMemPtr != 0);
REQUIRE(exampleMemPitch != 0);
REQUIRE(exampleSizeX != 0);
REQUIRE(exampleSizeY != 0);

// Access some element in a decomposed matrix.
// The macro uses multiplication. No X/Y range checking performed!
int value = *MATRIX_POINTER(example, 0, 0);

// Example element loop (not optimized):
uint32 sum = 0;

for (Space Y = 0; Y < exampleSizeY; ++Y)
    for (Space X = 0; X < exampleSizeX; ++X)
        sum += exampleMemPtr[X + Y * exampleMemPitch];

// Save rectangular area [10, 30) as a new matrix using
// "subs" (submatrix by size) function. Check that no clipping occured.
Matrix<const uint8> tmp1;
REQUIRE(example.subs(point(10), point(20), tmp1));

// Save rectangular area [10, 30) as a new matrix using
// "subr" (submatrix by rect) function. Check that no clipping occured.
Matrix<const uint8> tmp2;
REQUIRE(example.subr(point(10), point(30), tmp2));

// Remove const qualifier from element (avoid using it!)
Matrix<uint8> tmp3 = recastToNonConst(tmp2);

// Check that matrices have equal size.
REQUIRE(equalSize(example, tmp1, tmp2));
REQUIRE(equalSize(tmp1, tmp2, point(20)));

// Check that a matrix has non-zero size
REQUIRE(hasData(example));
REQUIRE(hasData(example.size()));

#endif

//================================================================
//
// MATRIX__CHECK_CONVERSION
//
//================================================================

template <typename SrcPointer, typename DstPointer>
struct Matrix_CheckConversion
{
    using Src = typename PtrElemType<SrcPointer>::T;
    using Dst = typename PtrElemType<DstPointer>::T;

    static const bool result = TYPE_EQUAL(Src, Dst) || TYPE_EQUAL(const Src, Dst);
};

//----------------------------------------------------------------

#define MATRIX__CHECK_CONVERSION(SrcPointer, DstPointer) \
    COMPILE_ASSERT((Matrix_CheckConversion<SrcPointer, DstPointer>::result))

//================================================================
//
// MATRIX_MUL_COORDS
//
//================================================================

#define MATRIX_MUL_COORDS(X, Y) \
    ((X) * (Y))

//================================================================
//
// MATRIX_POINTER
//
//================================================================

#define MATRIX_POINTER(matrix, X, Y) \
    (matrix##MemPtr + (X) + MATRIX_MUL_COORDS(Y, matrix##MemPitch))

#define MATRIX_POINTER_(matrix, pos) \
    MATRIX_POINTER(matrix, (pos).X, (pos).Y)

//----------------------------------------------------------------

#define MATRIX_ELEMENT(matrix, X, Y) \
    (*MATRIX_POINTER(matrix, X, Y))

#define MATRIX_ELEMENT_(matrix, pos) \
    MATRIX_ELEMENT(matrix, (pos).X, (pos).Y)

//================================================================
//
// MatrixPreconditions
//
// Static assertion:
//
// (1) sizeX >= 0 && sizeY >= 0
// (2) sizeX <= |pitch|
// (3) (sizeX * pitch * sizeof(*memPtr)) fits into Space type
//
//================================================================

class MatrixPreconditions
{
    sysinline MatrixPreconditions() {}
    friend sysinline MatrixPreconditions matrixPreconditionsAreVerified();
};

sysinline MatrixPreconditions matrixPreconditionsAreVerified()
    {return MatrixPreconditions();}

//================================================================
//
// matrixParamsAreValid
//
// Checks MatrixPreconditions.
//
//================================================================

template <Space elemSize>
bool matrixParamsAreValid(Space sizeX, Space sizeY, Space pitch);

//================================================================
//
// MatrixEx<Type>
//
// Supports any pointer type.
//
//================================================================

template <typename Pointer>
class MatrixEx
{

private:

    // Base pointer. If the matrix is empty, can be 0.
    Pointer theMemPtr;

    // Pitch. Can be negative. |pitch| >= sizeX.
    // If the matrix is empty, can be undefined.
    Space theMemPitch;

    // Dimensions. Always >= 0.
    // sizeX >= |pitch|
    Point<Space> theSize;

public:

    friend sysinline void exchange(MatrixEx<Pointer>& A, MatrixEx<Pointer>& B)
    {
        exchange(A.theMemPtr, B.theMemPtr);
        exchange(A.theMemPitch, B.theMemPitch);
        exchange(A.theSize, B.theSize);
    }

public:

    template <typename OtherPointer>
    friend class MatrixEx;

    using Type = typename PtrElemType<Pointer>::T;

    //
    // Create empty.
    // Create from 0 literal.
    //

public:

    struct TmpType {};

    sysinline MatrixEx(const TmpType* = 0)
        {assignNull();}

    //
    // Create uninitialized
    //

    sysinline MatrixEx(const ConstructUnitialized&)
        {}

    //
    // Create by parameters.
    //

    template <typename Ptr>
    sysinline MatrixEx(Ptr memPtr, Space memPitch, Space sizeX, Space sizeY)
        {assign(memPtr, memPitch, sizeX, sizeY);} // checked

    template <typename Ptr>
    sysinline MatrixEx(Ptr memPtr, Space memPitch, Space sizeX, Space sizeY, const MatrixPreconditions& preconditions)
        {assign(memPtr, memPitch, sizeX, sizeY, preconditions);} // unchecked, static assertion

    //
    // Create by an array.
    //

    template <typename OtherPointer>
    sysinline MatrixEx(const ArrayEx<OtherPointer>& that)
        :
        theMemPtr(that.thePtr),
        theMemPitch(that.theSize),
        theSize(point(that.theSize, 1))
    {
        MATRIX__CHECK_CONVERSION(OtherPointer, Pointer);
    }

    //
    // Export cast (no code generated, reinterpret pointer).
    //

    template <typename OtherPointer>
    sysinline operator const MatrixEx<OtherPointer>& () const
    {
        MATRIX__CHECK_CONVERSION(Pointer, OtherPointer);
        COMPILE_ASSERT(sizeof(MatrixEx<Pointer>) == sizeof(MatrixEx<OtherPointer>));
        return * (const MatrixEx<OtherPointer> *) this;
    }

    //
    // Assign data (checked).
    //

    sysinline bool assign(Pointer memPtr, Space memPitch, Space sizeX, Space sizeY)
    {
        bool ok = matrixParamsAreValid<sizeof(Type)>(sizeX, sizeY, memPitch);

        if_not (ok)
            {sizeX = 0; sizeY = 0;}

        theSize.X = sizeX;
        theSize.Y = sizeY;
        theMemPtr = memPtr;
        theMemPitch = memPitch;

        return ok;
    }

    //
    // Assign data (unchecked, static assertion).
    //

    sysinline void assign(Pointer memPtr, Space memPitch, Space sizeX, Space sizeY, const MatrixPreconditions&)
    {
        theMemPtr = memPtr;
        theMemPitch = memPitch;
        theSize.X = sizeX;
        theSize.Y = sizeY;
    }

#ifdef DBGPTR_MODE

    bool assign(ArrayPtr(Type) memPtr, Space memPitch, Space sizeX, Space sizeY)
    {
        assignNull();
        require(memPitch >= 0);

        ////

        Space area = 0;

        if (sizeY >= 1)
        {
            Space lastRow = 0;
            require(safeMul(memPitch, sizeY-1, lastRow));
            area = lastRow + sizeX;
        }

        return assign(unsafePtr(memPtr, area), memPitch, sizeX, sizeY);
    }

#endif

    ////

#ifdef DBGPTR_MODE

    bool assign(MatrixPtr(Type) memPtr, Space memPitch, Space sizeX, Space sizeY)
        {return assign(unsafePtr(memPtr, sizeX, sizeY), memPitch, sizeX, sizeY);}

    bool assign(MatrixPtr(Type) memPtr, Space memPitch, Space sizeX, Space sizeY, const MatrixPreconditions&)
        {return assign(unsafePtr(memPtr, sizeX, sizeY), memPitch, sizeX, sizeY);}

#endif

    //
    // Assign empty
    //

    sysinline void assignNull()
    {
        theMemPtr = Pointer(0);
        theMemPitch = 0;
        theSize.X = 0;
        theSize.Y = 0;
    }

    sysinline void assignEmptyFast()
    {
        theSize.Y = 0;
    }

    //
    // Get size.
    // Always >= 0.
    //

    sysinline Space sizeX() const
        {return theSize.X;}

    sysinline Space sizeY() const
        {return theSize.Y;}

    sysinline const Point<Space>& size() const
        {return theSize;}

    //
    // Get pitch.
    //

    sysinline Space memPitch() const
        {return theMemPitch;}

    //
    // Get base pointer.
    //

    sysinline Pointer memPtrUnsafeInternalUseOnly() const
        {return theMemPtr;}

#ifdef DBGPTR_MODE

    sysinline typename MatrixPtr(Type) memPtr() const
        {return MatrixPtrCreate(Type, theMemPtr, theMemPitch, theSize.X, theSize.Y, DbgptrMatrixPreconditions());}

#else

    sysinline Pointer memPtr() const
        {return theMemPtr;}

#endif

    //
    // subRow
    //

    template <typename OtherPointer>
    sysinline bool subRow(Space Y, ArrayEx<OtherPointer>& result) const
    {
        MATRIX__CHECK_CONVERSION(Pointer, OtherPointer);

        bool tmpOk = false;
        Pointer tmpPtr = Pointer(0);
        Space tmpSizeX = 0; // >= 0

        ////

        if (SpaceU(Y) < SpaceU(theSize.Y))
        {
            tmpPtr = MATRIX_POINTER(the, 0, Y);
            tmpSizeX = theSize.X; // >= 0
            tmpOk = true;
        }

        result.assign(tmpPtr, tmpSizeX, arrayPreconditionsAreVerified());
        return tmpOk;
    }

    //
    // subr
    //
    // Cuts a rectangular area from matrix, given by origin and end: point RIGHT AFTER the last element.
    // If the area is larger than the matrix, it is clipped to fit the matrix and false is returned.
    //

    template <typename OtherPointer>
    sysinline bool subr(Space orgX, Space orgY, Space endX, Space endY, MatrixEx<OtherPointer>& result) const
    {
        MATRIX__CHECK_CONVERSION(Pointer, OtherPointer);

        Space clOrgX = clampRange(orgX, 0, theSize.X);
        Space clOrgY = clampRange(orgY, 0, theSize.Y);

        Space clEndX = clampRange(endX, clOrgX, theSize.X);
        Space clEndY = clampRange(endY, clOrgY, theSize.Y);

        result.theMemPtr = MATRIX_POINTER(the, clOrgX, clOrgY);
        result.theMemPitch = theMemPitch;

        result.theSize.X = clEndX - clOrgX;
        result.theSize.Y = clEndY - clOrgY;

        return
            (clOrgX == orgX) &&
            (clOrgY == orgY) &&
            (clEndX == endX) &&
            (clEndY == endY);
    }

    ////

    template <typename Coord, typename OtherPointer>
    sysinline bool subr(const Point<Coord>& org, const Point<Coord>& end, MatrixEx<OtherPointer>& result) const
    {
        return subr(org.X, org.Y, end.X, end.Y, result);
    }

    //
    // subsUnsafeVeryDangerous
    //

    template <typename OtherPointer>
    sysinline void subsUnsafeVeryDangerous(Space orgX, Space orgY, Space sizeX, Space sizeY, MatrixEx<OtherPointer>& result) const
    {
        MATRIX__CHECK_CONVERSION(Pointer, OtherPointer);

        result.theMemPtr = MATRIX_POINTER(the, orgX, orgY);
        result.theMemPitch = theMemPitch;

        result.theSize.X = sizeX;
        result.theSize.Y = sizeY;
    }

    ////

    template <typename Coord, typename OtherPointer>
    sysinline void subsUnsafeVeryDangerous(const Point<Coord>& org, const Point<Coord>& size, MatrixEx<OtherPointer>& result) const
    {
        return subsUnsafeVeryDangerous(org.X, org.Y, size.X, size.Y, result);
    }

    //
    // subs
    //
    // Cuts a rectangular area from matrix, given by starting point and size.
    //
    // If the area is larger than the matrix, it is clipped to fit the matrix
    // and false is returned.
    //

    template <typename OtherPointer>
    sysinline bool subs(Space orgX, Space orgY, Space sizeX, Space sizeY, MatrixEx<OtherPointer>& result) const
    {
        MATRIX__CHECK_CONVERSION(Pointer, OtherPointer);

        Space clOrgX = clampRange(orgX, 0, theSize.X);
        Space clOrgY = clampRange(orgY, 0, theSize.Y);

        Space clSizeX = clampRange(sizeX, 0, theSize.X - clOrgX);
        Space clSizeY = clampRange(sizeY, 0, theSize.Y - clOrgY);

        result.theMemPtr = MATRIX_POINTER(the, clOrgX, clOrgY);
        result.theMemPitch = theMemPitch;

        result.theSize.X = clSizeX;
        result.theSize.Y = clSizeY;

        return
            (clOrgX == orgX) &&
            (clOrgY == orgY) &&
            (clSizeX == sizeX) &&
            (clSizeY == sizeY);
    }

    ////

    template <typename Coord, typename OtherPointer>
    sysinline bool subs(const Point<Coord>& org, const Point<Coord>& size, MatrixEx<OtherPointer>& result) const
    {
        return subs(org.X, org.Y, size.X, size.Y, result);
    }

    //
    // asArray
    //
    // Tries to export the matrix as array.
    // This is possible only for dense row allocation (pitch == size.X)
    //

    template <typename OtherPointer>
    sysinline bool asArray(Array<OtherPointer>& result) const
    {
        MATRIX__CHECK_CONVERSION(Pointer, OtherPointer);

        bool ok = true;

        check_flag(theMemPitch == theSize.X, ok);

        Space totalSize = theSize.X * theSize.Y;

        if_not (ok)
            totalSize = 0;

        result.assign(theMemPtr, totalSize, arrayPreconditionsAreVerified());

        return ok;
    }

};

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

template <typename Pointer>
sysinline bool matrixValidAccess(const MatrixEx<Pointer>& matrix, const Point<Space>& pos)
{
    return matrixValidAccess(matrix.size(), pos);
}

//================================================================
//
// Matrix
//
// Matrix for C++ address space: identical to MatrixEx<Type*>.
//
//================================================================

template <typename Type>
class Matrix : public MatrixEx<Type*>
{

public:

    using Base = MatrixEx<Type*>;

    UseType_(Base, TmpType);

    //
    // Constructors
    //

    sysinline Matrix(const TmpType* = 0)
        {}

    sysinline Matrix(Type* memPtr, Space memPitch, Space sizeX, Space sizeY)
        : Base(memPtr, memPitch, sizeX, sizeY) {}

    sysinline Matrix(Type* memPtr, Space memPitch, Space sizeX, Space sizeY, const MatrixPreconditions& preconditions)
        : Base(memPtr, memPitch, sizeX, sizeY, preconditions) {}

#ifdef DBGPTR_MODE

    sysinline Matrix(ArrayPtr(Type) memPtr, Space memPitch, Space sizeX, Space sizeY)
        : Base(memPtr, memPitch, sizeX, sizeY) {}

#endif

    sysinline Matrix(const Base& base)
        : Base(base) {}

    template <typename OtherPointer>
    sysinline Matrix(const Array<OtherPointer>& that)
        : Base(that) {}

    //
    // Export cast (no code generated, reinterpret pointer).
    //

    template <typename OtherType>
    sysinline operator const Matrix<OtherType>& () const
    {
        MATRIX__CHECK_CONVERSION(Type*, OtherType*);
        COMPILE_ASSERT(sizeof(Matrix<Type>) == sizeof(Matrix<OtherType>));
        return * (const Matrix<OtherType> *) this;
    }

    template <typename OtherType>
    sysinline operator const Matrix<OtherType> () const
    {
        MATRIX__CHECK_CONVERSION(Type*, OtherType*);
        COMPILE_ASSERT(sizeof(Matrix<Type>) == sizeof(Matrix<OtherType>));
        return * (const Matrix<OtherType> *) this;
    }

public:

    friend sysinline void exchange(Matrix<Type>& A, Matrix<Type>& B)
    {
        Base& baseA = A;
        Base& baseB = B;
        exchange(baseA, baseB);
    }

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
    auto prefix##SizeY = (matrix).sizeY()

//----------------------------------------------------------------

#define MATRIX_EXPOSE(matrix) \
    MATRIX_EXPOSE_GENERIC(matrix, matrix)

#define MATRIX_EXPOSE_EX(matrix, prefix) \
    MATRIX_EXPOSE_GENERIC(matrix, prefix)

//----------------------------------------------------------------

#define MATRIX_EXPOSE_UNSAFE(matrix, prefix) \
    auto prefix##MemPtr = (matrix).memPtrUnsafeInternalUseOnly(); \
    auto prefix##MemPitch = (matrix).memPitch(); \
    auto prefix##SizeX = (matrix).sizeX(); \
    auto prefix##SizeY = (matrix).sizeY()

//================================================================
//
// makeConst (fast)
//
//================================================================

template <typename Type>
sysinline const MatrixEx<const Type*>& makeConst(const MatrixEx<Type*>& matrix)
{
    COMPILE_ASSERT(sizeof(MatrixEx<const Type*>) == sizeof(MatrixEx<Type*>));
    return * (const MatrixEx<const Type*>*) &matrix;
}

//================================================================
//
// recastToNonConst
//
// Removes const qualifier from elements.
// Avoid using it!
//
//================================================================

template <typename Type>
sysinline const Matrix<Type>& recastToNonConst(const Matrix<const Type>& matrix)
{
    COMPILE_ASSERT(sizeof(Matrix<const Type>) == sizeof(Matrix<Type>));
    return * (const Matrix<Type>*) &matrix;
}

//================================================================
//
// equalSize support
//
//================================================================

template <>
GET_SIZE_DEFINE(Point<Space>, value)

template <typename Pointer>
GET_SIZE_DEFINE(MatrixEx<Pointer>, value.size())

template <typename Type>
GET_SIZE_DEFINE(Matrix<Type>, value.size())

template <typename Pointer>
sysinline Space getLayerCount(const MatrixEx<Pointer>& matrix)
    {return 1;}

//================================================================
//
// hasData
//
//================================================================

template <typename Type>
sysinline bool hasData(const MatrixEx<Type>& matrix)
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
