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

template <typename Pointer, typename Kit>
inline stdbool getMatrixMemoryRangeAsArray(const MatrixEx<Pointer>& img, ArrayEx<Pointer>& result, stdPars(Kit))
{
    MATRIX_EXPOSE_UNSAFE(img);

    result.assignNull();
    REQUIRE(imgMemPitch >= imgSizeX);

    result.assign(imgMemPtr, imgMemPitch * imgSizeY);

    returnTrue;
}

//================================================================
//
// copyMatrixAsArray
//
//================================================================

template <typename SrcPtr, typename DstPtr, typename Kit>
stdbool copyMatrixAsArray(const MatrixEx<SrcPtr>& srcMatrix, const MatrixEx<DstPtr>& dstMatrix, GpuCopyThunk& gpuCopy, stdPars(Kit))
{
    if_not (kit.dataProcessing)
        returnTrue;

    ////

    MatrixEx<SrcPtr> src = srcMatrix;
    MatrixEx<DstPtr> dst = dstMatrix;

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
    require(getMatrixMemoryRangeAsArray(src, srcArray, stdPass));

    ArrayEx<DstPtr> dstArray;
    require(getMatrixMemoryRangeAsArray(dst, dstArray, stdPass));

    ////

    require(gpuCopy(srcArray, dstArray, stdPass));

    returnTrue;
}
