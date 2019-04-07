#include "gpuImageConsoleAt.h"

#include "copyMatrixAsArray.h"
#include "dataAlloc/gpuMatrixMemory.h"
#include "dataAlloc/matrixMemory.h"

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// GpuBaseAtConsoleThunk
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// AtProviderFromGpuImage::saveImage
//
//================================================================

bool AtProviderFromGpuImage::saveImage(const Matrix<uint8_x4>& dest, stdNullPars)
{
    stdBegin;

    GpuMatrix<const uint8_x4> src = gpuImage;

    GpuCopyThunk copyToAtBuffer;

    if (src.memPitch() == dest.memPitch())
        require(copyMatrixAsArray(src, dest, copyToAtBuffer, stdPass));
    else
    {
        if (src.memPitch() >= 0 && dest.memPitch() >= 0)
            require(copyToAtBuffer(src, dest, stdPass));
        else if (src.memPitch() < 0 && dest.memPitch() < 0)
            require(copyToAtBuffer(flipMatrix(src), flipMatrix(dest), stdPass));
        else
            REQUIRE(false);
    }

    stdEnd;
}

//================================================================
//
// GpuBaseAtConsoleThunk::addImageCopyImpl
//
//================================================================

template <typename Type>
bool GpuBaseAtConsoleThunk::addImageCopyImpl(const GpuMatrix<const Type>& gpuMatrix, const ImgOutputHint& hint, stdNullPars)
{
    stdBegin;

    //
    // Allocate CPU matrix
    //

    MatrixMemory<Type> cpuMatrixMemory;
    require(cpuMatrixMemory.reallocForGpuExch(gpuMatrix.size(), stdPass));
    Matrix<Type> cpuMatrix = cpuMatrixMemory;

    //
    // Copy the matrix to CPU
    //

    {
        GpuCopyThunk gpuCopy;

        if (gpuMatrix.memPitch() >= 0)
            require(copyMatrixAsArray(gpuMatrix, cpuMatrixMemory, gpuCopy, stdPass));
        else
        {
            require(copyMatrixAsArray(flipMatrix(gpuMatrix), cpuMatrixMemory, gpuCopy, stdPass));
            cpuMatrix = flipMatrix(cpuMatrixMemory);
        }
    }

    //
    // Output the CPU matrix
    //

    if (kit.dataProcessing)
    {
        stdEnter;

        require(atImgConsole.addImage(cpuMatrix, hint, stdPass));
    }

    stdEnd;
}

//----------------------------------------------------------------

INSTANTIATE_FUNC(GpuBaseAtConsoleThunk::addImageCopyImpl<uint8>)
INSTANTIATE_FUNC(GpuBaseAtConsoleThunk::addImageCopyImpl<uint8_x4>)

//================================================================
//
// GpuBaseAtConsoleThunk::overlaySetImageBgr
//
//================================================================

bool GpuBaseAtConsoleThunk::overlaySetImageBgr(const Point<Space>& size, const GpuImageProviderBgr32& img, const ImgOutputHint& hint, stdNullPars)
{
    stdBegin;

    ////
  
    GPU_MATRIX_ALLOC(gpuImageMemory, uint8_x4, size);
    GpuMatrix<uint8_x4> gpuImage = flipMatrix(gpuImageMemory);
    require(img.saveImage(gpuImage, stdPass));

    ////

    if (kit.dataProcessing)
    {
        AtProviderFromGpuImage imageProvider(gpuImage, kit);
        require(atVideoOverlay.setImage(size, imageProvider, hint.desc, hint.id, textEnabled, stdPass));
    }

    overlaySet = true;

    ////

    stdEnd;
}
