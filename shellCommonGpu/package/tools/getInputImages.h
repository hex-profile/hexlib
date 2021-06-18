#pragma once

#include "gpuModuleHeader.h"
#include "dataAlloc/gpuArrayMemory.h"

namespace packageImpl {

//================================================================
//
// GetInputImages
//
// Allocates and computes input images.
//
//================================================================

template <typename Pixel>
struct GetInputImages
{
    virtual stdbool getImageSizes
    (
        const Array<Point<Space>>& imageSizes,
        stdPars(GpuModuleProcessKit)
    )
    =0;

    virtual stdbool getImages
    (
        const Array<GpuArrayMemory<Pixel>>& memories,
        const Array<GpuMatrix<const Pixel>>& images,
        stdPars(GpuModuleProcessKit)
    )
    =0;

    ////

    inline stdbool getImage
    (
        GpuArrayMemory<Pixel>& memory,
        GpuMatrix<const Pixel>& image,
        stdPars(GpuModuleProcessKit)
    )
    {
        return getImages(makeArray(&memory, 1), makeArray(&image, 1), stdPassThru);
    }
};

//----------------------------------------------------------------

}
