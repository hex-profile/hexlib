#include "gpuBaseConsoleByCpu.h"

#include "copyMatrixAsArray.h"
#include "dataAlloc/gpuMatrixMemory.h"
#include "dataAlloc/matrixMemory.h"
#include "dataAlloc/gpuArrayMemory.h"
#include "gpuMatrixCopy/gpuMatrixCopy.h"
#include "errorLog/debugBreak.h"
#include "gpuMatrixSet/gpuMatrixSet.h"
#include "data/spacex.h"

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// GpuBaseConsoleByCpuThunk
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// GpuBaseImageProvider::setImage
//
//================================================================

stdbool GpuBaseImageProvider::setImage(const GpuMatrix<const uint8_x4>& image, stdNullPars)
{
    Space absPitch = absv(image.memPitch());
    REQUIRE(image.sizeX() <= absPitch);

    ////

    constexpr Space maxAtRowByteAlignment = 64;
    constexpr Space alignment = maxAtRowByteAlignment / Space(sizeof(uint8_x4));
    COMPILE_ASSERT(alignment * sizeof(uint8_x4) == maxAtRowByteAlignment);

    COMPILE_ASSERT(alignment >= 1 && COMPILE_IS_POWER2(alignment));
    require(safeAdd(absPitch, alignment - 1, absPitch));
    absPitch = absPitch / alignment * alignment;

    ////

    require(buffer.realloc(absPitch * image.sizeY(), stdPass));
    gpuImage = image;

    returnTrue;
}

//================================================================
//
// GpuBaseImageProvider::saveImage
//
//================================================================

stdbool GpuBaseImageProvider::saveImage(const Matrix<uint8_x4>& dest, stdNullPars)
{
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

        //
        // Rearrange on GPU, not very fast, but it's still 
        // much faster than any manipulation on CPU.
        //

        require(gpuMatrixCopy(src, srcProper, stdPass));

        ////

        require(copyMatrixAsArray(srcProper, dst, copyToAtBuffer, stdPass));
    }

    returnTrue;
}

//================================================================
//
// GpuBaseConsoleByCpuThunk::addImageCopyImpl
//
//================================================================

template <typename Type>
stdbool GpuBaseConsoleByCpuThunk::addImageCopyImpl(const GpuMatrix<const Type>& gpuMatrix, const ImgOutputHint& hint, stdNullPars)
{
    if (hint.target == ImgOutputOverlay)
    {
        GpuBaseImageProvider imageProvider(kit);
        require(imageProvider.setImage(gpuMatrix, stdPass));

        require(baseVideoOverlay.setImage(gpuMatrix.size(), kit.dataProcessing, imageProvider, hint.desc, hint.id, textEnabled, stdPass));
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
            require(baseImageConsole.addImage(cpuMatrix, hint, stdPass));
        }
    }

    returnTrue;
}

//----------------------------------------------------------------

INSTANTIATE_FUNC(GpuBaseConsoleByCpuThunk::addImageCopyImpl<uint8_x4>)

//================================================================
//
// GpuBaseConsoleByCpuThunk::overlaySetImageBgr
//
//================================================================

stdbool GpuBaseConsoleByCpuThunk::overlaySetImageBgr(const Point<Space>& size, const GpuImageProviderBgr32& img, const ImgOutputHint& hint, stdNullPars)
{
    GPU_MATRIX_ALLOC(gpuImageMemory, uint8_x4, size);
    GpuMatrix<uint8_x4> gpuImage = flipMatrix(gpuImageMemory);
    require(img.saveImage(gpuImage, stdPass));

    ////

    GpuBaseImageProvider imageProvider(kit);
    require(imageProvider.setImage(gpuImage, stdPass));

    require(baseVideoOverlay.setImage(size, kit.dataProcessing, imageProvider, hint.desc, hint.id, textEnabled, stdPass));

    overlaySet = true;

    ////

    returnTrue;
}
