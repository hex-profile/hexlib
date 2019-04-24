#pragma once

#include "gpuProcessHeader.h"

//================================================================
//
// gpuMatrixCopy
//
// Not for using in highly optimized code.
//
//================================================================

template <typename Type>
stdbool gpuMatrixCopy(const GpuMatrix<const Type>& src, const GpuMatrix<Type>& dst, stdPars(GpuProcessKit));
