#pragma once

#include "gpuProcessHeader.h"

//================================================================
//
// gpuMatrixSet
//
// Not for using in highly optimized code.
//
//================================================================

template <typename Type>
stdbool gpuMatrixSet(const GpuMatrix<Type>& dst, const Type& value, stdPars(GpuProcessKit));
