#include "picPatternTest.h"

#include "cfgTools/boolSwitch.h"
#include "cfgTools/multiSwitch.h"
#include "dataAlloc/arrayObjectMemory.h"
#include "dataAlloc/gpuMatrixMemory.h"
#include "gaussSincResampling/resampleOneAndQuarter/downsampleOneAndQuarter.h"
#include "gaussSincResampling/resampleOneAndQuarter/upsampleOneAndQuarter.h"
#include "gpuMatrixSet/gpuMatrixSet.h"
#include "imageConsole/gpuImageConsole.h"
#include "numbers/mathIntrinsics.h"
#include "pyramid/gpuPyramidMemory.h"
#include "pyramid/pyramidScale.h"
#include "rndgen/rndgenBase.h"
#include "rndgen/rndgenMix.h"
#include "tests/preciseIntrinsicsCalib/patternGeneration/picPatternGeneration.h"
#include "userOutput/paramMsg.h"
#include "dataAlloc/gpuArrayMemory.h"
#include "dataAlloc/arrayMemory.h"
#include "userOutput/printMsgEx.h"
#include "gpuMatrixCopy/gpuMatrixCopy.h"

namespace picPatternTest {

using namespace picPatternGeneration;

//================================================================
//
// PicPatternTestImpl
//
//================================================================

class PicPatternTestImpl : public PicPatternTest
{

public:

    void serialize(const ModuleSerializeKit& kit);

    bool active() const
        {return displaySwitch != Display::Nothing;}

    void process(stdPars(GpuModuleProcessKit));

private:

    enum class Display {Nothing, Pattern, COUNT};
    ExclusiveMultiSwitch<Display, Display::COUNT, 0x23C2C032u> displaySwitch;

    ////

    NumericVar<uint32> randomSeed{0, typeMax<uint32>(), 0xCAFECAFE};

    ////

    NumericVar<Point<Space>> imageSize{point(0), point(typeMax<Space>()), point(3840, 2160)};

    ////

    NumericVar<Space> pyramidLevelsCfg{1, typeMax<Space>(), 30};

    ////

    static constexpr auto pyramidRounding = RoundUp;
    static constexpr float32 pyramidFactor = 5.f / 4;
    PyramidScaleArray pyramidScale{pyramidFactor};

    ////

    // Additional factor used after normalization the pyramid band levels amplitude
    NumericVar<float32> normalizationFactor{0, typeMax<float32>(), 4};

};

//----------------------------------------------------------------

UniquePtr<PicPatternTest> PicPatternTest::create()
{
    return makeUnique<PicPatternTestImpl>();
}

//================================================================
//
// PicPatternTestImpl::serialize
//
//================================================================

void PicPatternTestImpl::serialize(const ModuleSerializeKit& kit)
{
    displaySwitch.serialize
    (
        kit,
        STR("Display"),
        {STR("<Nothing>")},
        {STR("Pattern"), STR("Ctrl+Alt+I")}
    );

    randomSeed.serialize(kit, STR("Random Seed"));
    imageSize.serialize(kit, STR("Image Size"));
    pyramidLevelsCfg.serialize(kit, STR("Pyramid Levels"));

    normalizationFactor.serialize(kit, STR("Normalization Factor"), STR(""),
        STR("Additional factor used after normalization the pyramid band levels amplitude"));
}

//================================================================
//
// PicPatternTestImpl::process
//
//================================================================

void PicPatternTestImpl::process(stdPars(GpuModuleProcessKit))
{
    if_not (active())
        return;

    ////

    auto pyramidLevels = pyramidLevelsCfg();

    int displayedScale = kit.display.scaleIndex(0, pyramidLevels-1);

    //----------------------------------------------------------------
    //
    // Pyramid.
    //
    //----------------------------------------------------------------

    GpuPyramidMemory<float32> pattern;

    pattern.realloc(imageSize, pyramidLevels, 1, pyramidScale, pyramidRounding, {}, stdPass);

    //----------------------------------------------------------------
    //
    // Fill the bottom level of the pyramid with random values.
    //
    //----------------------------------------------------------------

    auto rndgen = randomSeed();

    REQUIRE(pyramidLevels >= 1);
    genRandomMatrix(pattern[0], rndgen, stdPass);

    //----------------------------------------------------------------
    //
    // Resamplers.
    //
    //----------------------------------------------------------------

    auto downsample = gaussSincResampling::downsampleOneAndQuarterConservative<float32, float32, float32>;
    auto upsample = gaussSincResampling::upsampleOneAndQuarterBalanced<float32, float32, float32>;

    //----------------------------------------------------------------
    //
    // Downsampling of the pyramid from bottom to top.
    //
    //----------------------------------------------------------------

    for_range (s, 1, pyramidLevels)
        {downsample(pattern[s - 1], pattern[s], BORDER_MIRROR, stdPass);}

    //----------------------------------------------------------------
    //
    // Dividing into bands by scale.
    //
    //----------------------------------------------------------------

    GpuPyramidMemory<float32> bands;
    bands.realloc(imageSize, pyramidLevels, 1, pyramidScale, pyramidRounding, {}, stdPass);

    ////

    for_count (s, pyramidLevels-1)
    {
        upsample(pattern[s+1], bands[s], BORDER_MIRROR, stdPass);

        combineLinearly(pattern[s], bands[s], bands[s], 1, -1, stdPass);
    }

    ////

    gpuMatrixCopy(pattern[pyramidLevels-1], bands[pyramidLevels-1], stdPass);

    //----------------------------------------------------------------
    //
    // Count statistics and normalize band levels energy.
    //
    //----------------------------------------------------------------

    for_count (s, pyramidLevels)
    {
        auto image = pattern[s];

        ////

        GPU_ARRAY_ALLOC(sumAbsGpu, float32, 1);
        gpuArraySet(sumAbsGpu, 0.f, stdPass);

        GPU_ARRAY_ALLOC(sumSqGpu, float32, 1);
        gpuArraySet(sumSqGpu, 0.f, stdPass);

        ////

        computeStats(image, sumAbsGpu, sumSqGpu, stdPass);

        ////

        auto divCount = fastRecip(float32(areaOf(image)));

        ARRAY_ALLOC_FOR_GPU_EXCH(sumAbs, float32, 1);
        ARRAY_ALLOC_FOR_GPU_EXCH(sumSq, float32, 1);

        {
            GpuCopyThunk gpuCopy;
            gpuCopy(sumAbsGpu, sumAbs, stdPass);
            gpuCopy(sumSqGpu, sumSq, stdPass);
        }

        ////

        auto avgAbs = divCount * (kit.dataProcessing ? sumAbs[0] : 0.f);
        auto avgSq = divCount * (kit.dataProcessing ? sumAbs[0] : 0.f);
        avgSq = fastSqrt(avgSq);

        ////

        auto factor = 1.f / (avgSq * normalizationFactor);

        if (s == displayedScale)
        {
            printMsgL(kit, STR("Avg abs %"), fltf(avgAbs, 3));
            printMsgL(kit, STR("Rms %"), fltf(avgSq, 3));
            printMsgL(kit, STR("Factor %"), fltf(factor, 3));
        }

        if (s == pyramidLevels-1) // The last level is low-pass
            continue;

        ////

        if (kit.alternative)
            combineLinearly(bands[s], bands[s], bands[s], factor, 0, stdPass);
    }

    //----------------------------------------------------------------
    //
    // Reconstruct the pyramid from the bands.
    //
    //----------------------------------------------------------------

    GpuPyramidMemory<float32> reconstruction;
    reconstruction.realloc(imageSize, pyramidLevels, 1, pyramidScale, pyramidRounding, {}, stdPass);

    ////

    gpuMatrixCopy(bands[pyramidLevels-1], reconstruction[pyramidLevels-1], stdPass);

    ////

    for_range_reverse (s, 0, pyramidLevels-1)
    {
        upsample(reconstruction[s+1], reconstruction[s], BORDER_MIRROR, stdPass);

        combineLinearly(reconstruction[s], bands[s], reconstruction[s], 1, 1, stdPass);
    }

    //----------------------------------------------------------------
    //
    // Output of one level of the original or reconstructed pyramid.
    //
    //----------------------------------------------------------------

    int stage = kit.display.temporalIndex(0, 2);

    auto displayedImage =
        stage == 0 ? pattern[displayedScale] :
        stage == 1 ? bands[displayedScale] :
        reconstruction[displayedScale];


    ////

    kit.gpuImageConsole.addMatrixEx
    (
        displayedImage,
        -kit.display.factor,
        +kit.display.factor,
        point(pyramidScale(displayedScale)),
        INTERP_NEAREST,
        imageSize,
        BORDER_ZERO,
        paramMsg(STR("Level %"), displayedScale),
        stdPass
    );
}

//----------------------------------------------------------------

}
