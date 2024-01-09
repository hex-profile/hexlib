#pragma once

#include "gpuProcessHeader.h"

//================================================================
//
// gpuMatrixCopy
//
//================================================================

template <typename Type>
stdbool gpuMatrixCopyImpl(const GpuMatrixAP<const Type>& src, const GpuMatrixAP<Type>& dst, stdPars(GpuProcessKit));

////

template <typename SrcType, typename SrcPitch, typename DstType, typename DstPitch>
sysinline stdbool gpuMatrixCopy(const GpuMatrix<SrcType, SrcPitch>& src, const GpuMatrix<DstType, DstPitch>& dst, stdPars(GpuProcessKit))
    {return gpuMatrixCopyImpl<DstType>(src, dst, stdPassThru);}

//================================================================
//
// gpuArrayCopy
//
//================================================================

template <typename Type>
stdbool gpuArrayCopy(const GpuArray<const Type>& src, const GpuArray<Type>& dst, stdPars(GpuProcessKit));
