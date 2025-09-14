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
void gpuMatrixSetImpl(const GpuMatrixAP<Type>& dst, const Type& value, stdPars(GpuProcessKit));

template <typename Type, typename Pitch>
sysinline void gpuMatrixSet(const GpuMatrix<Type, Pitch>& dst, const Type& value, stdPars(GpuProcessKit))
    {gpuMatrixSetImpl<Type>(dst, value, stdPassThru);}

//================================================================
//
// gpuArraySet
//
//================================================================

template <typename Type>
void gpuArraySet(const GpuArray<Type>& dst, const Type& value, stdPars(GpuProcessKit));
