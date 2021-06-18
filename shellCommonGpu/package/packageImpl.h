#pragma once

#include "data/array.h"
#include "data/matrix.h"

namespace packageImpl {

//================================================================
//
// uncast
//
//================================================================

template <typename Type>
inline Array<Type> uncast(const ArrayBase<Type>& arr)
{
    return recastEqualLayout<Array<Type>>(arr);
}

//----------------------------------------------------------------

template <typename Type>
inline Matrix<Type> uncast(const MatrixBase<Type>& mat)
{
    COMPILE_ASSERT_EQUAL_LAYOUT(MatrixBase<Type>, Matrix<Type>);
    return recastEqualLayout<Matrix<Type>>(mat);
}

//----------------------------------------------------------------

template <typename Type>
inline Array<const Matrix<Type>> uncast(const ArrayBase<const MatrixBase<Type>>& arr)
{
    COMPILE_ASSERT_EQUAL_LAYOUT(MatrixBase<Type>, Matrix<Type>);
    return recastEqualLayout<Array<const Matrix<Type>>>(arr);
}

//----------------------------------------------------------------

}
