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
    return recastEqualLayout<Matrix<Type>>(mat);
}

//================================================================
//
// uncastRef
//
// Only because of GCC bugs.
//
//================================================================

template <typename Type>
inline const Array<Type>& uncastRef(const ArrayBase<Type>& arr)
{
    return recastEqualLayout<Array<Type>>(arr);
}

//----------------------------------------------------------------

template <typename Type>
inline const Matrix<Type>& uncastRef(const MatrixBase<Type>& mat)
{
    return recastEqualLayout<Matrix<Type>>(mat);
}

//----------------------------------------------------------------

}
