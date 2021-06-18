#pragma once

#include "package/tools/getInputImages.h"
#include "extLib/types/vectorBase.h"

namespace packageImpl {

//================================================================
//
// GetInputImagesFromCpu
//
//================================================================

template <typename Pixel>
class GetInputImagesFromCpu : public GetInputImages<Pixel>
{

public:

    GetInputImagesFromCpu(const Array<const Matrix<const Pixel>>& cpuImages)
        : cpuImages{cpuImages} {}

    GetInputImagesFromCpu(const Matrix<const Pixel>& cpuImage)
        : cpuImages{makeArray(&cpuImage, 1)} {}

public:

    virtual stdbool getImageSizes
    (
        const Array<Point<Space>>& imageSizes,
        stdPars(GpuModuleProcessKit)
    );

    virtual stdbool getImages
    (
        const Array<GpuArrayMemory<Pixel>>& memories,
        const Array<GpuMatrix<const Pixel>>& images,
        stdPars(GpuModuleProcessKit)
    );

private:

    Array<const Matrix<const Pixel>> cpuImages;

};

//================================================================
//
// GetMonoImages
//
//================================================================

class GetMonoImages : public GetInputImages<uint8>
{

public:

    static constexpr Space maxImageCount = 4;
    using MonoPixel = uint8;
    using ColorPixel = uint8_x4;

public:

    GetMonoImages(GetInputImages<ColorPixel>& getColorImages)
        : getColorImages(getColorImages) {}

public:

    virtual stdbool getImageSizes
    (
        const Array<Point<Space>>& imageSizes,
        stdPars(GpuModuleProcessKit)
    )
    {
        return getColorImages.getImageSizes(imageSizes, stdPassThru);
    }

public:

    virtual stdbool getImages
    (
        const Array<GpuArrayMemory<MonoPixel>>& memories,
        const Array<GpuMatrix<const MonoPixel>>& images,
        stdPars(GpuModuleProcessKit)
    );

private:

    GetInputImages<ColorPixel>& getColorImages;

};

//----------------------------------------------------------------

}
