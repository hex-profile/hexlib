#include "gpuBaseConsoleByCpu.h"

#include "colorConversions/colorConversions.h"
#include "copyMatrixAsArray.h"
#include "data/spacex.h"
#include "dataAlloc/gpuArrayMemory.h"
#include "dataAlloc/gpuMatrixMemory.h"
#include "dataAlloc/matrixMemory.h"
#include "errorLog/debugBreak.h"
#include "gpuMatrixCopy/gpuMatrixCopy.h"
#include "gpuMatrixSet/gpuMatrixSet.h"

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
// Preallocates the intermediate GPU buffer assuming maximum
// expected CPU row byte alignment.
//
//================================================================

stdbool GpuBaseImageProvider::setImage(const GpuMatrixAP<const ColorPixel>& image, stdParsNull)
{
    Space absPitch = absv(image.memPitch());
    REQUIRE(image.sizeX() <= absPitch);

    ////

    constexpr Space maxExpectedCpuRowByteAlignment = 64;
    constexpr Space alignment = maxExpectedCpuRowByteAlignment / Space(sizeof(ColorPixel));
    COMPILE_ASSERT(alignment * sizeof(ColorPixel) == maxExpectedCpuRowByteAlignment);

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
// GpuBaseImageProvider::saveBgr32
//
// * If the destination pitch is equal to the source pitch, copies it as an array.
//
// * If the destination pitch is not equal to the source one,
// transforms the image on GPU to the destination pitch
// using the preallocated intermediate buffer.
//
// * Doesn't do dynamic allocations based on the destination pitch.
//
//================================================================

stdbool GpuBaseImageProvider::saveBgr32(const MatrixAP<ColorPixel>& dest, stdParsNull)
{
    auto src = gpuImage;
    auto dst = dest;

    ////

    GpuCopyThunk gpuCopy;

    ////

    if (src.memPitch() == dst.memPitch())
    {

        require(copyMatrixAsArray(src, dst, gpuCopy, stdPass));

    }
    else
    {
        REQUIRE(src.size() == dst.size());

        ////

        Space absPitch = absv(dst.memPitch());

        ARRAY_EXPOSE(buffer);
        REQUIRE(dest.sizeX() <= absPitch);
        REQUIRE(absPitch * dest.sizeY() <= bufferSize);

        GpuMatrixAP<ColorPixel> srcProper;
        REQUIRE(srcProper.assignValidated(bufferPtr, absPitch, dest.sizeX(), dest.sizeY()));

        ////

        if (dst.memPitch() < 0)
            srcProper = flipMatrix(srcProper);

        //
        // Rearrange on GPU, not very fast, but it's still
        // much faster than any manipulation on CPU.
        //

        require(gpuMatrixCopy(src, srcProper, stdPass));

        ////

        require(copyMatrixAsArray(srcProper, dst, gpuCopy, stdPass));
    }

    returnTrue;
}

//================================================================
//
// GpuBaseImageProvider::saveBgr24
//
//================================================================

stdbool GpuBaseImageProvider::saveBgr24(const MatrixAP<MonoPixel>& dest, stdParsNull)
{
    auto src = gpuImage;
    auto dst = dest;

    REQUIRE(src.sizeX() * 3 == dst.sizeX());
    REQUIRE(src.sizeY() == dst.sizeY());

    ////

    GpuCopyThunk gpuCopy;

    ////

    Space absPitch = absv(dst.memPitch());
    REQUIRE(dest.sizeX() <= absPitch);

    ////

    ARRAY_EXPOSE_UNSAFE_EX(buffer, bufferColor);
    COMPILE_ASSERT(sizeof(ColorPixel) % sizeof(MonoPixel) == 0);

    using GpuMonoPtr = GpuPtr(MonoPixel);

    auto bufferPtr = GpuMonoPtr(bufferColorPtr);
    auto bufferSize = bufferColorSize * Space{sizeof(ColorPixel) / sizeof(MonoPixel)};

    ////

    REQUIRE(absPitch * dest.sizeY() <= bufferSize);

    GpuMatrixAP<MonoPixel> srcProper;
    REQUIRE(srcProper.assignValidated(bufferPtr, absPitch, dest.sizeX(), dest.sizeY()));

    ////

    if (dst.memPitch() < 0)
        srcProper = flipMatrix(srcProper);

    //
    // Rearrange on GPU, not very fast, but it's still
    // much faster than any manipulation on CPU.
    //

    require(convertBgr32ToBgr24(src, srcProper, stdPass));

    ////

    require(copyMatrixAsArray(srcProper, dst, gpuCopy, stdPass));

    ////

    returnTrue;
}

//================================================================
//
// GpuBaseConsoleByCpuThunk::addImageCopyImpl
//
//================================================================

template <typename Type>
stdbool GpuBaseConsoleByCpuThunk::addImageCopyImpl(const GpuMatrixAP<const Type>& gpuMatrix, const ImgOutputHint& hint, stdPars(Kit))
{
    if (hint.target == ImgOutputOverlay)
    {
        GpuBaseImageProvider imageProvider(kit);
        require(imageProvider.setImage(gpuMatrix, stdPass));

        require(baseVideoOverlay.overlaySet(gpuMatrix.size(), kit.dataProcessing, imageProvider, hint.desc, hint.id, textEnabled, stdPass));
    }
    else
    {
        //
        // Allocate CPU matrix
        //

        MATRIX_ALLOC_FOR_GPU_EXCH(cpuMatrixMemory, Type, gpuMatrix.size());
        auto cpuMatrix = relaxToAnyPitch(cpuMatrixMemory);

        //
        // Copy the matrix to CPU
        //

        {
            GpuCopyThunk gpuCopy;

            if (gpuMatrix.memPitch() >= 0)
            {
                require(copyMatrixAsArray(gpuMatrix, cpuMatrixMemory, gpuCopy, stdPass));
            }
            else
            {
                require(copyMatrixAsArray(flipMatrix(gpuMatrix), cpuMatrixMemory, gpuCopy, stdPass));
                cpuMatrix = flipMatrix(cpuMatrixMemory());
            }
        }

        //
        // Output the CPU matrix
        //

        {
            stdEnter;
            require(baseImageConsole.addImage(cpuMatrix, hint, kit.dataProcessing, stdPass));
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

stdbool GpuBaseConsoleByCpuThunk::overlaySetImageBgr(const Point<Space>& size, const GpuImageProviderBgr32& img, const ImgOutputHint& hint, stdPars(Kit))
{
    GPU_MATRIX_ALLOC(gpuImageMemory, uint8_x4, size);
    auto gpuImage = flipMatrix(gpuImageMemory);
    require(img.saveImage(gpuImage, stdPass));

    ////

    GpuBaseImageProvider imageProvider(kit);
    require(imageProvider.setImage(gpuImage, stdPass));

    require(baseVideoOverlay.overlaySet(size, kit.dataProcessing, imageProvider, hint.desc, hint.id, textEnabled, stdPass));

    ////

    returnTrue;
}
