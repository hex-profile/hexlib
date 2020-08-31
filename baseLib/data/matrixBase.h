#pragma once

#include <type_traits>

#include "data/arrayBase.h"
#include "types/pointTypes.h"

//================================================================
//
// MatrixBase<Type>
//
// MatrixBase memory layout: base pointer, pitch and dimensions.
//
//----------------------------------------------------------------
//
// memPtr:
// Points to (0, 0) element. Can be undefined if the matrix is empty.
//
// memPitch:
// The difference of pointers to (X, Y+1) and (X, Y) elements.
// The difference is expressed in elements (not bytes) and can be negative.
//
// sizeX, sizeY:
// The width and height of the matrix. Both are >= 0.
// If either of them is zero, the matrix is empty.
//
//================================================================


//================================================================
//
// matrixBaseIsValid
//
//================================================================

template <Space elemSize>
bool matrixBaseIsValid(Space sizeX, Space sizeY, Space pitch);

//================================================================
//
// MatrixBase
//
// MatrixBase for C++ address space: identical to MatrixEx<Type*>.
//
//================================================================

template <typename Type>
class MatrixBase
{

    template <typename OtherType>
    friend class MatrixBase;

public:

    //
    // Empty.
    //

    HEXLIB_INLINE MatrixBase()
    {
        assignNull();
    }

    //
    // Create by parameters.
    //

    HEXLIB_INLINE MatrixBase(Type* memPtr, Space memPitch, Space sizeX, Space sizeY)
    {
        assign(memPtr, memPitch, sizeX, sizeY);
    }

    //
    // Create by an array.
    //

    template <typename OtherType>
    HEXLIB_INLINE MatrixBase(const ArrayBase<OtherType>& that)
        :
        theMemPtrUnsafe(that.thePtr),
        theMemPitch(that.theSize),
        theSize(point(that.theSize, 1))
    {
        static_assert(CheckConversion<OtherType, Type>::value, "");
    }

    //
    // Export cast (no code generated).
    //

    template <typename OtherType>
    HEXLIB_INLINE operator const MatrixBase<OtherType>& () const
    {
        static_assert(CheckConversion<Type, OtherType>::value, "");
        static_assert(sizeof(MatrixBase<Type>) == sizeof(MatrixBase<OtherType>), "");
        static_assert(alignof(MatrixBase<Type>) == alignof(MatrixBase<OtherType>), "");
        return * (const MatrixBase<OtherType> *) this;
    }

    //
    // Export const
    //

    template <typename OtherType>
    HEXLIB_INLINE operator MatrixBase<OtherType> () const
    {
        static_assert(CheckConversion<Type, OtherType>::value, "");
        static_assert(sizeof(MatrixBase<Type>) == sizeof(MatrixBase<OtherType>), "");
        static_assert(alignof(MatrixBase<Type>) == alignof(MatrixBase<OtherType>), "");
        return * (const MatrixBase<OtherType> *) this;
    }

    //
    // Assign by parameters (checked).
    //

    HEXLIB_INLINE bool assign(Type* memPtr, Space memPitch, Space sizeX, Space sizeY)
    {
        bool ok = matrixBaseIsValid<sizeof(Type)>(sizeX, sizeY, memPitch);

        if (!ok)
            {sizeX = 0; sizeY = 0;}

        theSize.X = sizeX;
        theSize.Y = sizeY;
        theMemPtrUnsafe = memPtr;
        theMemPitch = memPitch;

        return ok;
    }

    //
    // Assign empty
    //

    HEXLIB_INLINE void assignNull()
    {
        theMemPtrUnsafe = nullptr;
        theMemPitch = 0;
        theSize.X = 0;
        theSize.Y = 0;
    }

    //
    // Get size. Always >= 0.
    //

    HEXLIB_INLINE Space sizeX() const
        {return theSize.X;}

    HEXLIB_INLINE Space sizeY() const
        {return theSize.Y;}

    HEXLIB_INLINE const Point<Space>& size() const
        {return theSize;}

    //
    // Get ptr and pitch.
    //

    HEXLIB_INLINE Space memPitch() const
        {return theMemPitch;}

    HEXLIB_INLINE Type* memPtr() const
        {return theMemPtrUnsafe;}

private:

    template <typename Src, typename Dst>
    struct CheckConversion
    {
        static constexpr bool value = std::is_convertible<Src, Dst>::value;
    };

private:

    // Base pointer. If the matrix is empty, can be nullptr.
    Type* theMemPtrUnsafe;

    // Pitch. Can be negative. |pitch| >= sizeX.
    // If the matrix is empty, can be undefined.
    Space theMemPitch;

    // Dimensions. Always >= 0.
    // sizeX >= |pitch|
    Point<Space> theSize;

};

//================================================================
//
// makeConst (fast)
//
//================================================================

template <typename Type>
HEXLIB_INLINE const MatrixBase<const Type>& makeConst(const MatrixBase<Type>& matrix)
{
    static_assert(sizeof(MatrixBase<const Type>) == sizeof(MatrixBase<Type>), "");
    static_assert(alignof(MatrixBase<const Type>) == alignof(MatrixBase<Type>), "");

    return * (const MatrixBase<const Type>*) &matrix;
}

//================================================================
//
// hasData
//
//================================================================

template <typename Type>
HEXLIB_INLINE bool hasData(const MatrixBase<Type>& matrix)
{
    return matrix.sizeX() >= 1 && matrix.sizeY() >= 1;
}
