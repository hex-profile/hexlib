#include "getInputImagesImpl.h"

#include "colorConversions/colorConversions.h"
#include "dataAlloc/arrayMemoryStatic.h"
#include "dataAlloc/arrayObjectMemoryStatic.h"
#include "dataAlloc/gpuMatrixMemory.h"
#include "flipMatrix.h"
#include "gpuAppliedApi/gpuAppliedApi.h"

namespace packageImpl {

//================================================================
//
// GetInputImagesFromCpu<Pixel>::getImageSizes
//
//================================================================

template <typename Pixel>
stdbool GetInputImagesFromCpu<Pixel>::getImageSizes
(
    const Array<Point<Space>>& imageSizes,
    stdPars(GpuModuleProcessKit)
)
{
    REQUIRE(equalSize(cpuImages, imageSizes));
    int imageCount = cpuImages.size();

    for_count (k, imageCount)
        imageSizes[k] = cpuImages[k].size();

    returnTrue;
}

//================================================================
//
// GetInputImagesFromCpu<Pixel>::getImages
//
//================================================================

template <typename Pixel>
stdbool GetInputImagesFromCpu<Pixel>::getImages
(
    const Array<GpuArrayMemory<Pixel>>& memories,
    const Array<GpuMatrix<const Pixel>>& images,
    stdPars(GpuModuleProcessKit)
)
{
    REQUIRE(equalSize(cpuImages, memories, images));

    int imageCount = images.size();

    ////

    GpuCopyThunk copyCpuBuffers; // Wait before releasing CPU buffers.

    ////

    for_count (k, imageCount)
    {
        auto src = cpuImages[k];
        auto& dst = images[k];

        ////

        bool inverted = false;

        if (src.memPitch() < 0)
            {src = flipMatrix(src); inverted = true;}

        ////

        MATRIX_EXPOSE_UNSAFE(src);

        Array<const Pixel> srcArray;
        REQUIRE(srcMemPitch >= srcSizeX);
        srcArray.assign(srcMemPtr, srcMemPitch * srcSizeY);

        auto& dstArray = memories[k];
        require(dstArray.realloc(srcArray.size(), stdPass));

        ////

        require(copyCpuBuffers(srcArray, dstArray, stdPass));

        ////

        REQUIRE(dst.assign(dstArray.ptr(), srcMemPitch, srcSizeX, srcSizeY));

        if (inverted)
            dst = flipMatrix(dst);
    }

    returnTrue;
}

//----------------------------------------------------------------

template class GetInputImagesFromCpu<uint8>;
template class GetInputImagesFromCpu<uint8_x4>;

//================================================================
//
// GetMonoImages::getImages
//
//================================================================

stdbool GetMonoImages::getImages
(
    const Array<GpuArrayMemory<MonoPixel>>& memories,
    const Array<GpuMatrix<const MonoPixel>>& images,
    stdPars(GpuModuleProcessKit)
)
{
    REQUIRE(equalSize(memories, images));
    int imageCount = images.size();

    ////

    ARRAY_OBJECT_STATIC_ALLOC(imageSizes, Point<Space>, maxImageCount, imageCount);
    require(getColorImages.getImageSizes(imageSizes, stdPass));

    ////

    ARRAY_OBJECT_STATIC_ALLOC(dest, GpuMatrix<MonoPixel>, maxImageCount, imageCount);

    ////

    for_count (k, images.size())
    {
        auto size = imageSizes[k];
        Space memPitch = 0;

        {
            GPU_MATRIX_ALLOC(tmp, MonoPixel, size);
            memPitch = tmp.memPitch();
        }

        REQUIRE(memPitch >= 0);
        require(memories[k].realloc(memPitch * size.Y, stdPass));

        REQUIRE(dest[k].assign(memories[k].ptr(), memPitch, size.X, size.Y));
        images[k] = dest[k];
    }

    ////

    ARRAY_OBJECT_STATIC_ALLOC(colorMemories, GpuArrayMemory<ColorPixel>, maxImageCount, imageCount);
    ARRAY_OBJECT_STATIC_ALLOC(colorImages, GpuMatrix<const ColorPixel>, maxImageCount, imageCount);

    require(getColorImages.getImages(colorMemories, colorImages, stdPass));

    ////

    for_count (k, images.size())
        require(convertBgr32ToMono(colorImages[k], dest[k], stdPass));

    ////

    returnTrue;
}

//----------------------------------------------------------------

}
