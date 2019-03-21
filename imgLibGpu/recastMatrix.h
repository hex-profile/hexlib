#pragma once

#include "data/matrix.h"
#include "data/gpuMatrix.h"

//================================================================
//
// recastMatrix
//
//================================================================

template <typename Dst, typename Src>
inline Matrix<Dst> recastMatrix(const Matrix<Src>& src)
{
    MATRIX_EXPOSE(src);

    COMPILE_ASSERT(sizeof(Src) == sizeof(Dst));

    return Matrix<Dst>
    (
        (Dst*) unsafePtr(srcMemPtr, srcSizeX, srcSizeY),
        srcMemPitch, srcSizeX, srcSizeY
    );
}

//================================================================
//
// recastMatrix
//
//================================================================

template <typename Dst, typename Src>
inline GpuMatrix<Dst> recastMatrix(const GpuMatrix<Src>& src)
{
    MATRIX_EXPOSE(src);

    COMPILE_ASSERT(sizeof(Src) == sizeof(Dst));

    using SrcPtr = GpuPtr(Src);
    using DstPtr = GpuPtr(Dst);

    DstPtr dstPtr = DstPtr(GpuAddrU(unsafePtr(srcMemPtr, srcSizeX, srcSizeY)));

    return GpuMatrix<Dst>(dstPtr, srcMemPitch, srcSizeX, srcSizeY);
}
