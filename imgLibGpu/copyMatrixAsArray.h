#pragma once

#include <string.h>

#include "gpuAppliedApi/gpuAppliedApi.h"
#include "stdFunc/stdFunc.h"
#include "flipMatrix.h"

//================================================================
//
// getMatrixMemoryRangeAsArray
//
//================================================================

template <typename MatrixType, typename ArrayType, typename Kit>
inline void getMatrixMemoryRangeAsArray(const MatrixType& img, ArrayType& result, stdPars(Kit))
{
    MATRIX_EXPOSE_UNSAFE(img);

    result.assignNull();
    REQUIRE(imgMemPitch >= imgSizeX);

    result.assignUnsafe(imgMemPtr, imgMemPitch * imgSizeY);
}

//================================================================
//
// copyMatrixAsArray
//
//================================================================

template <typename SrcPtr, typename SrcPitch, typename DstPtr, typename DstPitch, typename Kit>
sysinline void copyMatrixAsArray
(
    const MatrixEx<SrcPtr, SrcPitch>& srcMatrix,
    const MatrixEx<DstPtr, DstPitch>& dstMatrix,
    GpuCopyThunk& gpuCopy,
    stdPars(Kit)
)
{
    if_not (kit.dataProcessing)
        return;

    ////

    MatrixEx<SrcPtr, PitchMayBeNegative> src = srcMatrix;
    MatrixEx<DstPtr, PitchMayBeNegative> dst = dstMatrix;

    ////

    REQUIRE(equalSize(src, dst));
    REQUIRE(src.memPitch() == dst.memPitch());

    ////

    if (src.memPitch() < 0)
    {
        src = flipMatrix(src);
        dst = flipMatrix(dst);
    }

    REQUIRE(src.memPitch() >= src.sizeX());

    ////

    ArrayEx<SrcPtr> srcArray;
    getMatrixMemoryRangeAsArray(src, srcArray, stdPass);

    ArrayEx<DstPtr> dstArray;
    getMatrixMemoryRangeAsArray(dst, dstArray, stdPass);

    ////

    gpuCopy(srcArray, dstArray, stdPass);
}
