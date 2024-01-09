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
stdbool gpuMatrixSetImpl(const GpuMatrixAP<Type>& dst, const Type& value, stdPars(GpuProcessKit));

template <typename Type, typename Pitch>
sysinline stdbool gpuMatrixSet(const GpuMatrix<Type, Pitch>& dst, const Type& value, stdPars(GpuProcessKit))
    {return gpuMatrixSetImpl<Type>(dst, value, stdPassThru);}

//================================================================
//
// gpuArraySet
//
//================================================================

template <typename Type>
stdbool gpuArraySet(const GpuArray<Type>& dst, const Type& value, stdPars(GpuProcessKit));
