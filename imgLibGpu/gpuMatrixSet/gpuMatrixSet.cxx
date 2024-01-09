#include "gpuMatrixSet.inl"

#include "vectorTypes/vectorOperations.h"

//================================================================
//
// gpuMatrixSet
//
//================================================================

VECTOR_INT_FOREACH(GPU_MATRIX_SET_DEFINE, _) \
VECTOR_FLOAT_FOREACH(GPU_MATRIX_SET_DEFINE, _)

//================================================================
//
// gpuArraySet
//
//================================================================

VECTOR_INT_FOREACH(GPU_ARRAY_SET_DEFINE, _) \
VECTOR_FLOAT_FOREACH(GPU_ARRAY_SET_DEFINE, _)
