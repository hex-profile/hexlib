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

    inline MatrixBase()
    {
        assignNull();
    }

    //
    // Create by parameters.
    //

    inline MatrixBase(Type* memPtr, Space memPitch, Space sizeX, Space sizeY)
    {
        assign(memPtr, memPitch, sizeX, sizeY);
    }

    //
    // Create by an array.
    //

    template <typename OtherType>
    inline MatrixBase(const ArrayBase<OtherType>& that)
        :
        theMemPtr(that.thePtr),
        theMemPitch(that.theSize),
        theSize(point(that.theSize, 1))
    {
        static_assert(CheckConversion<OtherType, Type>::value, "");
    }

    //
    // Export cast (no code generated).
    //

    template <typename OtherType>
    inline operator const MatrixBase<OtherType>& () const
    {
        static_assert(CheckConversion<Type, OtherType>::value, "");
        static_assert(sizeof(MatrixBase<Type>) == sizeof(MatrixBase<OtherType>));
        static_assert(alignof(MatrixBase<Type>) == alignof(MatrixBase<OtherType>));
        return * (const MatrixBase<OtherType> *) this;
    }

    //
    // Assign by parameters (checked).
    //

    inline bool assign(Type* memPtr, Space memPitch, Space sizeX, Space sizeY)
    {
        bool ok = matrixBaseIsValid<sizeof(Type)>(sizeX, sizeY, memPitch);

        if (!ok)
            {sizeX = 0; sizeY = 0;}

        theSize.X = sizeX;
        theSize.Y = sizeY;
        theMemPtr = memPtr;
        theMemPitch = memPitch;

        return ok;
    }

    //
    // Assign empty
    //

    inline void assignNull()
    {
        theMemPtr = nullptr;
        theMemPitch = 0;
        theSize.X = 0;
        theSize.Y = 0;
    }

    //
    // Get size. Always >= 0.
    //

    inline Space sizeX() const
        {return theSize.X;}

    inline Space sizeY() const
        {return theSize.Y;}

    inline const Point<Space>& size() const
        {return theSize;}

    //
    // Get ptr and pitch.
    //

    inline Space memPitch() const
        {return theMemPitch;}

    inline Type* memPtr() const
        {return theMemPtr;}

private:

    template <typename Src, typename Dst>
    struct CheckConversion
    {
        static constexpr bool value = std::is_same<Src, Dst>::value || std::is_same<const Src, Dst>::value;
    };

private:

    // Base pointer. If the matrix is empty, can be nullptr.
    Type* theMemPtr;

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
inline const MatrixBase<const Type>& makeConst(const MatrixBase<Type>& matrix)
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
inline bool hasData(const MatrixBase<Type>& matrix)
{
    return matrix.sizeX() >= 1 && matrix.sizeY() >= 1;
}
