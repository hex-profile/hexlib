#pragma once

#include "gpuProcessHeader.h"

//================================================================
//
// gpuMatrixCopy
//
//================================================================

template <typename Type>
void gpuMatrixCopyImpl(const GpuMatrixAP<const Type>& src, const GpuMatrixAP<Type>& dst, stdPars(GpuProcessKit));

////

template <typename SrcType, typename SrcPitch, typename DstType, typename DstPitch>
sysinline void gpuMatrixCopy(const GpuMatrix<SrcType, SrcPitch>& src, const GpuMatrix<DstType, DstPitch>& dst, stdPars(GpuProcessKit))
    {gpuMatrixCopyImpl<DstType>(src, dst, stdPassThru);}

//================================================================
//
// gpuArrayCopy
//
//================================================================

template <typename Type>
void gpuArrayCopy(const GpuArray<const Type>& src, const GpuArray<Type>& dst, stdPars(GpuProcessKit));
