#pragma once

#include "gpuProcessHeader.h"

//================================================================
//
// GPU_MATRIX_SET_FOREACH
//
//================================================================

#define GPU_MATRIX_SET_FOREACH(action, extra) \
    VECTOR_INT_FOREACH(action, extra) \
    VECTOR_FLOAT_FOREACH(action, extra) \

//================================================================
//
// gpuMatrixSetFunc
//
//================================================================

template <typename Type>
inline bool gpuMatrixSetFunc(const GpuMatrix<Type>& dst, const Type& value, stdPars(GpuProcessKit));

template <typename Type, typename Matrix>
inline bool gpuMatrixSet(const Matrix& dst, const Type& value, stdPars(GpuProcessKit))
    {return gpuMatrixSetFunc<Type>(dst, value, stdPassThru);}

//================================================================
//
// gpuMatrixSetFunc_*
//
//================================================================

#define TMP_MACRO(Type, o) \
    \
    bool gpuMatrixSetFunc_##Type(const GpuMatrix<Type>& dst, const Type& value, stdPars(GpuProcessKit)); \
    \
    template <> \
    inline bool gpuMatrixSetFunc(const GpuMatrix<Type>& dst, const Type& value, stdPars(GpuProcessKit)) \
        {return gpuMatrixSetFunc_##Type(dst, value, stdPassThru);}

GPU_MATRIX_SET_FOREACH(TMP_MACRO, o)

#undef TMP_MACRO
