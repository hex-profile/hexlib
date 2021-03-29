#include "visualizeComplexFilter.h"

#include "dataAlloc/gpuMatrixMemory.h"
#include "errorLog/errorLog.h"
#include "gaussSincResampling/resampleTwice/upsampleTwice.h"
#include "gpuMatrixCopy/gpuMatrixCopy.h"
#include "imageConsole/gpuImageConsole.h"
#include "pyramid/gpuPyramidMemory.h"
#include "userOutput/paramMsg.h"
#include "visualizeComplexFilter/visualizeComplexFilterKernel.h"

namespace visualizeComplex {

//================================================================
//
// visualizeComplexFilter
//
//================================================================

stdbool visualizeComplexFilter
(
    const GpuMatrix<const ComplexFloat>& image,
    const Point<float32>& filterFreq, // In original resolution.
    float32 filterSubsamplingFactor,
    int displayedOrientation,
    int displayedScale,
    const Point<Space>& displayedSize,
    float32 displayMagnitude,
    float32 arrowFactor,
    const PyramidScale& pyramidScale,
    const FormatOutputAtom& name,
    stdPars(GpuModuleProcessKit)
)
{
    Space s = displayedScale;

    //----------------------------------------------------------------
    //
    // Pre-upsampling with high quality filter.
    //
    //----------------------------------------------------------------
                     
    GpuMatrixMemory<ComplexFloat> preUpsampleImage;

    int preUpsampleFactor = 1;

    if (kit.display.interpolation)
    {
        preUpsampleFactor = 2;

        auto preUpsampleFunc = gaussSincResampling::upsampleTwiceBalanced<ComplexFloat, ComplexFloat, ComplexFloat>;

        require(preUpsampleImage.realloc(image.size() * preUpsampleFactor, stdPass));
        require(preUpsampleFunc(image, preUpsampleImage, BORDER_MIRROR, stdPass));
    }

    //----------------------------------------------------------------
    //
    // Build final image and display it.
    //
    //----------------------------------------------------------------

    bool fullscreen = kit.display.fullscreen;

    auto computedSize = convertNearest<Space>(image.size() * filterSubsamplingFactor);
    GPU_MATRIX_ALLOC(upsampledImage, ComplexFloat, fullscreen ? displayedSize : computedSize);

    float32 upsampingFactor = filterSubsamplingFactor * pyramidScale(fullscreen ? s : 0);

    ////

    require
    (
        visualizeComplexFilterFunc
        (
            preUpsampleFactor != 1 ? preUpsampleImage : image,
            upsampledImage,
            point(upsampingFactor / preUpsampleFactor),
            kit.display.interpolation,
            kit.display.modulation,
            filterFreq / pyramidScale(fullscreen ? s : 0), // Freq in dst space.
            stdPass
        )
    );

    ////

    require
    (
        kit.gpuImageConsole.addVectorImage
        (
            upsampledImage,
            displayMagnitude * kit.display.factor,
            point(1.f), INTERP_NEAREST,
            displayedSize,
            BORDER_CLAMP,
            ImgOutputHint(paramMsg(displayedOrientation == -1 ? STR("%, Scale %") : STR("%, Scale %, Orient %"), name, s, displayedOrientation))
            .setArrowFactor(arrowFactor),
            stdPass
        )
    );

    ////

    returnTrue;
}

//----------------------------------------------------------------

}
