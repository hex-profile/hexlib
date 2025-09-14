#pragma once

#include "data/array.h"
#include "data/matrix.h"

namespace packageImpl {

//================================================================
//
// uncast<Array>
//
//================================================================

template <typename Type>
inline auto& uncast(const ArrayBase<Type>& arr)
{
    return recastEqualLayout<const Array<Type>>(arr);
}

//================================================================
//
// uncast<Matrix>
//
//================================================================

template <typename Type, typename Pitch>
inline auto& uncast(const MatrixBase<Type, Type*, Pitch>& mat)
{
    return recastEqualLayout<const Matrix<Type, Pitch>>(mat);
}

//----------------------------------------------------------------

}
