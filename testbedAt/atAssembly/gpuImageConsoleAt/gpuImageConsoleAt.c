#include "gpuImageConsoleAt.h"

#include "copyMatrixAsArray.h"
#include "dataAlloc/gpuMatrixMemory.h"
#include "dataAlloc/matrixMemory.h"
#include "dataAlloc/gpuArrayMemory.h"
#include "gpuMatrixCopy/gpuMatrixCopy.h"
#include "errorLog/debugBreak.h"

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
// AtProviderFromGpuImage::setImage
//
//================================================================

stdbool AtProviderFromGpuImage::setImage(const GpuMatrix<const uint8_x4>& image, stdNullPars)

{
    stdBegin;

    Space absPitch = absv(image.memPitch());
    REQUIRE(image.sizeX() <= absPitch);

    require(buffer.realloc(absPitch * image.sizeY(), stdPass));
    gpuImage = image;

    stdEnd;
}

//================================================================
//
// AtProviderFromGpuImage::saveImage
//
//================================================================

stdbool AtProviderFromGpuImage::saveImage(const Matrix<uint8_x4>& dest, stdNullPars)

{
    stdBegin;

    GpuMatrix<const uint8_x4> src = gpuImage;
    Matrix<uint8_x4> dst = dest;

    ////

    GpuCopyThunk copyToAtBuffer;

    ////

    if (src.memPitch() == dst.memPitch())
    {

        require(copyMatrixAsArray(src, dst, copyToAtBuffer, stdPass));

    }
    else
    {
        REQUIRE(src.size() == dst.size());

        ////

        Space absPitch = absv(dst.memPitch());

        ARRAY_EXPOSE(buffer);
        REQUIRE(dest.sizeX() <= absPitch);
        REQUIRE(absPitch * dest.sizeY() <= bufferSize);

        GpuMatrix<uint8_x4> srcProper(bufferPtr, absPitch, dest.sizeX(), dest.sizeY());

        ////

        if (dst.memPitch() < 0)
            srcProper = flipMatrix(srcProper);

        ////

        require(gpuMatrixCopy(src, srcProper, stdPass));

        ////

        require(copyMatrixAsArray(srcProper, dst, copyToAtBuffer, stdPass));
    }

    stdEnd;
}

//================================================================
//
// GpuBaseAtConsoleThunk::addImageCopyImpl
//
//================================================================

template <typename Type>
stdbool GpuBaseAtConsoleThunk::addImageCopyImpl(const GpuMatrix<const Type>& gpuMatrix, const ImgOutputHint& hint, stdNullPars)

{
    stdBegin;

    if (hint.target == ImgOutputOverlay)
    {
        AtProviderFromGpuImage imageProvider(kit);
        require(imageProvider.setImage(gpuMatrix, stdPass));

        if (kit.dataProcessing)
            require(atVideoOverlay.setImage(gpuMatrix.size(), imageProvider, hint.desc, hint.id, textEnabled, stdPass));

        overlaySet = true;
    }
    else
    {

        //
        // Allocate CPU matrix
        //

        MATRIX_ALLOC_FOR_GPU_EXCH(cpuMatrixMemory, Type, gpuMatrix.size());
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
    }

    stdEnd;
}

//----------------------------------------------------------------

INSTANTIATE_FUNC(GpuBaseAtConsoleThunk::addImageCopyImpl<uint8_x4>)

//================================================================
//
// GpuBaseAtConsoleThunk::overlaySetImageBgr
//
//================================================================

stdbool GpuBaseAtConsoleThunk::overlaySetImageBgr(const Point<Space>& size, const GpuImageProviderBgr32& img, const ImgOutputHint& hint, stdNullPars)

{
    stdBegin;

    ////
  
    GPU_MATRIX_ALLOC(gpuImageMemory, uint8_x4, size);
    GpuMatrix<uint8_x4> gpuImage = flipMatrix(gpuImageMemory);
    require(img.saveImage(gpuImage, stdPass));

    ////

    AtProviderFromGpuImage imageProvider(kit);
    require(imageProvider.setImage(gpuImage, stdPass));

    if (kit.dataProcessing)
        require(atVideoOverlay.setImage(size, imageProvider, hint.desc, hint.id, textEnabled, stdPass));

    overlaySet = true;

    ////

    stdEnd;
}
