#include "resamplingTest.h"

#include "numbers/float16/float16Type.h"
#include "gpuSupport/gpuTool.h"
#include "gpuDevice/loadstore/storeNorm.h"

#include "numbers/mathIntrinsics.h"
#include "numbers/mathIntrinsics.h"
#include "gpuSupport/gpuTexTools.h"
#include "gpuDevice/loadstore/loadNorm.h"
#include "vectorTypes/vectorOperations.h"
#include "readInterpolate/gpuTexCubic.h"

#if HOSTCODE
#include "bsplinePrefilter/bsplinePrefilter.h"
#include "cfgTools/multiSwitch.h"
#include "dataAlloc/gpuMatrixMemory.h"
#include "dataAlloc/matrixMemory.h"
#include "diagStatistics/diagStatistics.h"
#include "gaussSincResampling/gaussSincResamplingSettings.h"
#include "gaussSincResampling/resampleFourTimes/downsampleFourTimes.h"
#include "gaussSincResampling/resampleFourTimes/upsampleFourTimes.h"
#include "gaussSincResampling/resampleThreeTimes/upsampleThreeTimes.h"
#include "gaussSincResampling/resampleOneAndHalf/downsampleOneAndHalf.h"
#include "gaussSincResampling/resampleOneAndHalf/upsampleOneAndHalf.h"
#include "gaussSincResampling/resampleOneAndThird/downsampleOneAndThird.h"
#include "gaussSincResampling/resampleOneAndThird/upsampleOneAndThird.h"
#include "gaussSincResampling/resampleOneAndQuarter/downsampleOneAndQuarter.h"
#include "gaussSincResampling/resampleOneAndQuarter/upsampleOneAndQuarter.h"
#include "gaussSincResampling/resampleOne/downsampleOne.h"
#include "gaussSincResampling/resampleTwice/downsampleTwice.h"
#include "gaussSincResampling/resampleTwice/upsampleTwice.h"
#include "gpuMatrixCopy/gpuMatrixCopy.h"
#include "imageConsole/gpuImageConsole.h"
#include "storage/classThunks.h"
#include "userOutput/printMsgEx.h"
#include "userOutput/paramMsg.h"
#endif

namespace resamplingTest {

//================================================================
//
// FloatPixel
//
//================================================================

sysinline float32 pixelMin() {return -1;}
sysinline float32 pixelMax() {return +1;}

using FloatPixel = float16;

//================================================================
//
// sinc
//
//================================================================

template <typename Float>
sysinline Float sinc(Float x)
{
    Float t = Float(pi64) * x;
    Float result = fastDivide(sinf(t), t);
    if_not (def(result)) result = 1;
    return result;
}

//================================================================
//
// GaussSinc
//
//================================================================

class GaussSinc
{

public:

    sysinline GaussSinc() {}

    sysinline GaussSinc(float32 sigma, float32 theta, float32 filterRadius)
        : sigma(sigma), theta(theta), filterRadius(filterRadius) {}

    sysinline float32 radius() const {return filterRadius;}

    sysinline float32 func(float32 x) const
    {
        float32 value = (1/theta) * sinc((1/theta) * x) * expf((-0.5f/(theta*theta*sigma*sigma) ) * x * x);
        if_not (absv(x) <= filterRadius) value = 0;
        return value;
    }

private:

    float32 sigma;
    float32 theta;
    float32 filterRadius;

};

//================================================================
//
// GaussKernel
//
//================================================================

class GaussKernel
{

public:

    sysinline GaussKernel() {}

    sysinline GaussKernel(float32 sigma, float32 filterRadius)
        : sigma(sigma), filterRadius(filterRadius) {}

    sysinline float32 radius() const {return filterRadius;}

    sysinline float32 func(float32 x) const
    {
        float32 value = expf(-0.5f * square(x / sigma));
        if_not (absv(x) <= filterRadius) value = 0;
        return value;
    }

private:

    float32 sigma;
    float32 filterRadius;

};

//================================================================
//
// CubicKernel
//
//================================================================

class CubicKernel
{

public:

    sysinline CubicKernel() {}

    sysinline float32 radius() const {return 2.f;}

    sysinline float32 func(float32 x) const
    {
        x = absv(x);
        float32 x2 = x*x;
        float32 x3 = x*x*x;

        ////

        float32 result = 0;

        if (x < 2)
            result = -0.5f*x3 + 2.5f*x2 - 4*x + 2;

        if (x < 1)
            result = 1.5f*x3 - 2.5f*x2 + 1;

        ////

        return result;
    }

};

//================================================================
//
// resampleModel
//
//================================================================

#define FUNCNAME resamplePyramidModel
#define PIXEL FloatPixel
#define KERNEL GaussSinc
# include "resamplingTest.inl"
#undef PIXEL
#undef KERNEL
#undef FUNCNAME

#define FUNCNAME resampleGaussModel
#define PIXEL FloatPixel
#define KERNEL GaussKernel
# include "resamplingTest.inl"
#undef PIXEL
#undef KERNEL
#undef FUNCNAME

#define FUNCNAME resampleCubicModel
#define PIXEL FloatPixel
#define KERNEL CubicKernel
# include "resamplingTest.inl"
#undef PIXEL
#undef KERNEL
#undef FUNCNAME

//================================================================
//
// convertToFloatPixel
//
//================================================================

GPUTOOL_2D
(
    convertToFloatPixel,
    PREP_EMPTY,
    ((const uint8_x4, src))
    ((FloatPixel, dst)),
    PREP_EMPTY,
    {
        auto tmp = loadNorm(src);
        auto gray = (1.f/3) * (tmp.x + tmp.y + tmp.z);
        storeNorm(dst, 2 * gray - 1);
    }
)

GPUTOOL_2D
(
    convertToFloatPixelEx,
    PREP_EMPTY,
    ((const int8, src))
    ((FloatPixel, dst)),
    ((float32, factor)),
    storeNorm(dst, factor * loadNorm(src));
)

//================================================================
//
// computeError
//
//================================================================

GPUTOOL_2D
(
    computeError,
    PREP_EMPTY,
    ((const FloatPixel, src0))
    ((const FloatPixel, src1))
    ((float32, error)),
    PREP_EMPTY,
    storeNorm(error, loadNorm(src1) - loadNorm(src0));
)

//================================================================
//
// upsampleTexCubic
//
//================================================================

GPUTOOL_2D
(
    upsampleTexCubic,
    ((const FloatPixel, src, INTERP_NONE, BORDER_MIRROR)),
    ((FloatPixel, dst)),
    ((Point<float32>, dstToSrcFactor)),
    {
        Point<float32> srcPos = point(Xs, Ys) * dstToSrcFactor;
        storeNorm(dst, tex2DCubic(srcSampler, srcPos, srcTexstep));
    }
)

//================================================================
//
// upsampleTexCubicBspline
//
//================================================================

GPUTOOL_2D
(
    upsampleTexCubicBspline,
    ((const FloatPixel, src, INTERP_NONE, BORDER_MIRROR)),
    ((FloatPixel, dst)),
    ((Point<float32>, dstToSrcFactor)),
    {
        Point<float32> srcPos = point(Xs, Ys) * dstToSrcFactor;
        storeNorm(dst, tex2DCubicBspline(srcSampler, srcPos, srcTexstep));
    }
)

//================================================================
//
// upsampleTexCubicBsplineFast
//
//================================================================

GPUTOOL_2D
(
    upsampleTexCubicBsplineFast,
    ((const FloatPixel, src, INTERP_LINEAR, BORDER_MIRROR)),
    ((FloatPixel, dst)),
    ((Point<float32>, dstToSrcFactor)),
    {
        Point<float32> srcPos = point(Xs, Ys) * dstToSrcFactor;

        auto result = tex2DCubicBsplineFast(srcSampler, srcPos, srcTexstep);
        storeNorm(dst, result);
    }
)

//================================================================
//
// ResamplingTestImpl
//
//================================================================

#if HOSTCODE

class ResamplingTestImpl
{

public:

    void serialize(const ModuleSerializeKit& kit);
    bool active() const {return displaySwitch != DisplayNothing;}

    stdbool process(const ProcessParams& o, stdPars(GpuModuleProcessKit));

private:

    enum DisplayType {DisplayNothing, DisplaySource, DisplayDestination, DisplayError, DisplayCount};
    ExclusiveMultiSwitch<DisplayType, DisplayCount, 0x57296E2F> displaySwitch;
    NumericVarStatic<Space, 0, 1024, 0> errorSpatialMargin;
    NumericVar<float32> variableUpsampleFactor{1/16.f, 16.f, fastSqrt(2.f)};

};

//----------------------------------------------------------------

CLASSTHUNK_CONSTRUCT_DESTRUCT(ResamplingTest)
CLASSTHUNK_VOID1(ResamplingTest, serialize, const ModuleSerializeKit&)
CLASSTHUNK_BOOL_CONST0(ResamplingTest, active)
CLASSTHUNK_BOOL_STD1(ResamplingTest, process, const ProcessParams&, GpuModuleProcessKit)

//================================================================
//
// ResamplingTestImpl::serialize
//
//================================================================

void ResamplingTestImpl::serialize(const ModuleSerializeKit& kit)
{
    displaySwitch.serialize
    (
        kit, STR("Display"),
        {STR("<Nothing>"), STR("")},
        {STR("Resample Test: Source"), STR("Shift+Alt+1")},
        {STR("Resample Test: Destination"), STR("Shift+Alt+2")},
        {STR("Resample Test: Error"), STR("Shift+Alt+3")}
    );

    errorSpatialMargin.serialize(kit, STR("Error Spatial Margin"));
    variableUpsampleFactor.serialize(kit, STR("Variable Upsample Factor"));
}

//================================================================
//
// ResamplingTestImpl::process
//
//================================================================

stdbool ResamplingTestImpl::process(const ProcessParams& o, stdPars(GpuModuleProcessKit))
{
    DisplayType displayType = kit.verbosity >= Verbosity::On ? displaySwitch : DisplayNothing;

    if (displayType == DisplayNothing)
        returnTrue;

    //----------------------------------------------------------------
    //
    // Input
    //
    //----------------------------------------------------------------

    GPU_MATRIX_ALLOC(srcImage, FloatPixel, o.gpuRgbFrame.size());
    require(convertToFloatPixel(o.gpuRgbFrame, srcImage, stdPass));
    Point<Space> srcSize = srcImage.size();

    ////

    if (displayType == DisplaySource)
    {
        require(kit.gpuImageConsole.addMatrixEx(srcImage,
            kit.display.factor * pixelMin(), kit.display.factor * pixelMax(), point(1.f),
            INTERP_NEAREST, srcSize, BORDER_ZERO, STR("Source Image"), stdPass));
    }

    //----------------------------------------------------------------
    //
    // Downsample pyramid model
    //
    //----------------------------------------------------------------

    enum class Test
    {
        DownsampleFourTimes,
        DownsampleTwice,
        DownsampleOneAndHalf,
        DownsampleOneAndThird,
        DownsampleOneAndQuarter,
        DownsampleOne,
        UpsampleOneAndQuarter,
        UpsampleOneAndThird,
        UpsampleOneAndHalf,
        UpsampleTwice,
        UpsampleThreeTimes,
        UpsampleFourTimes,
        InterpolationBicubic,
        InterpolationUnserBspline,
        UnprefilterUnserBspline,
        COUNT
    };

    Test test = Test(kit.display.temporalIndex(0, int(Test::COUNT) - 1));

    float32 resampleFactorScalar =
        (test == Test::DownsampleTwice) ? 1/2.f :
        (test == Test::UpsampleTwice) ? 2.f :
        (test == Test::DownsampleOneAndHalf) ? 2.f / 3 :
        (test == Test::UpsampleOneAndHalf) ? 3.f / 2 :
        (test == Test::DownsampleOneAndThird) ? 3.f / 4 :
        (test == Test::UpsampleOneAndThird) ? 4.f / 3 :
        (test == Test::DownsampleOneAndQuarter) ? 4.f / 5 :
        (test == Test::UpsampleOneAndQuarter) ? 5.f / 4 :
        (test == Test::DownsampleOne) ? 1.f :
        (test == Test::DownsampleFourTimes) ? 1/4.f :
        (test == Test::UpsampleThreeTimes) ? 3.f :
        (test == Test::UpsampleFourTimes) ? 4.f :
        (test == Test::InterpolationBicubic) ? variableUpsampleFactor :
        (test == Test::InterpolationUnserBspline) ? variableUpsampleFactor :
        (test == Test::UnprefilterUnserBspline) ? 1.f :
        1.f;

    printMsgL(kit, STR("Test %0, scale factor %1"), int(test), resampleFactorScalar);

    Point<float32> resampleFactor = point(resampleFactorScalar);

    ////

    namespace gsr = gaussSincResampling;

    GaussSinc downsamplingKernel(gsr::sigma, gsr::conservativeTheta, 4 * gsr::sigma * gsr::conservativeTheta);
    GaussSinc upsamplingKernel(gsr::sigma, gsr::balancedTheta, 4 * gsr::sigma * gsr::balancedTheta);

    CubicKernel cubicKernel;

    ////

    Point<Space> dstSize = convertUp<Space>(convertFloat32(srcImage.size()) * resampleFactor);
    GPU_MATRIX_ALLOC(dstImage, FloatPixel, dstSize);

    if
    (
        test == Test::DownsampleOneAndHalf ||
        test == Test::UpsampleOneAndHalf ||
        test == Test::DownsampleOneAndThird ||
        test == Test::UpsampleOneAndThird ||
        test == Test::DownsampleOneAndQuarter ||
        test == Test::UpsampleOneAndQuarter ||
        test == Test::DownsampleOne ||
        test == Test::DownsampleTwice ||
        test == Test::UpsampleTwice ||
        test == Test::DownsampleFourTimes ||
        test == Test::UpsampleThreeTimes ||
        test == Test::UpsampleFourTimes
    )
        require(resamplePyramidModel(srcImage, dstImage, 1.f/resampleFactor, resampleFactorScalar <= 1 ? downsamplingKernel : upsamplingKernel, stdPass));

    if (test == Test::InterpolationBicubic)
        require(resampleCubicModel(srcImage, dstImage, 1.f/resampleFactor, cubicKernel, stdPass));

    if (test == Test::InterpolationUnserBspline)
    {
        GPU_MATRIX_ALLOC(srcImagePrefiltered, FloatPixel, srcSize);
        require((bsplineCubicPrefilter<FloatPixel, FloatPixel, FloatPixel>(srcImage, srcImagePrefiltered, point(1.f), BORDER_MIRROR, stdPass)));
        require(upsampleTexCubicBspline(srcImagePrefiltered, dstImage, 1.f/resampleFactor, stdPass));
    }

    if (test == Test::UnprefilterUnserBspline)
    {
        require(gpuMatrixCopy(makeConst(srcImage), dstImage, stdPass));
    }

    //----------------------------------------------------------------
    //
    // Result tested.
    //
    //----------------------------------------------------------------

    GPU_MATRIX_ALLOC(dstImageTest, FloatPixel, dstSize);

    using namespace gaussSincResampling;

    if (test == Test::DownsampleOneAndHalf)
        require((downsampleOneAndHalfConservative<FloatPixel, FloatPixel, FloatPixel>(makeConst(srcImage), dstImageTest, BORDER_MIRROR, stdPass)));

    if (test == Test::UpsampleOneAndHalf)
        require((upsampleOneAndHalfBalanced<FloatPixel, FloatPixel, FloatPixel>(makeConst(srcImage), dstImageTest, BORDER_MIRROR, stdPass)));

    ////

    if (test == Test::DownsampleOneAndThird)
        require((downsampleOneAndThirdConservative<FloatPixel, FloatPixel, FloatPixel>(makeConst(srcImage), dstImageTest, BORDER_MIRROR, stdPass)));

    if (test == Test::UpsampleOneAndThird)
        require((upsampleOneAndThirdBalanced<FloatPixel, FloatPixel, FloatPixel>(makeConst(srcImage), dstImageTest, BORDER_MIRROR, stdPass)));

    ////

    if (test == Test::DownsampleOneAndQuarter)
        require((downsampleOneAndQuarterConservative<FloatPixel, FloatPixel, FloatPixel>(makeConst(srcImage), dstImageTest, BORDER_MIRROR, stdPass)));

    if (test == Test::UpsampleOneAndQuarter)
        require((upsampleOneAndQuarterBalanced<FloatPixel, FloatPixel, FloatPixel>(makeConst(srcImage), dstImageTest, BORDER_MIRROR, stdPass)));

    ////

    if (test == Test::DownsampleOne)
        require((downsampleOneConservative<FloatPixel, FloatPixel, FloatPixel>(makeConst(srcImage), dstImageTest, BORDER_MIRROR, stdPass)));

    ////

    if (test == Test::DownsampleTwice)
        require((downsampleTwiceConservative<FloatPixel, FloatPixel, FloatPixel>(makeConst(srcImage), dstImageTest, BORDER_MIRROR, stdPass)));

    if (test == Test::UpsampleTwice)
        require((upsampleTwiceBalanced<FloatPixel, FloatPixel, FloatPixel>(makeConst(srcImage), dstImageTest, BORDER_MIRROR, stdPass)));

    ////

    if (test == Test::UpsampleThreeTimes)
        require((upsampleThreeTimesBalanced<FloatPixel, FloatPixel, FloatPixel>(makeConst(srcImage), dstImageTest, BORDER_MIRROR, stdPass)));

    ////

    if (test == Test::DownsampleFourTimes)
        require((downsampleFourTimesConservative<FloatPixel, FloatPixel, FloatPixel>(makeConst(srcImage), dstImageTest, BORDER_MIRROR, stdPass)));

    if (test == Test::UpsampleFourTimes)
        require((upsampleFourTimesBalanced<FloatPixel, FloatPixel, FloatPixel>(makeConst(srcImage), dstImageTest, BORDER_MIRROR, stdPass)));

    ////

    if (test == Test::InterpolationBicubic)
        require(upsampleTexCubic(srcImage, dstImageTest, 1.f/resampleFactor, stdPass));

    ////

    if (test == Test::InterpolationUnserBspline)
    {
        GPU_MATRIX_ALLOC(srcImagePrefiltered, FloatPixel, srcSize);
        require((bsplineCubicPrefilter<FloatPixel, FloatPixel, FloatPixel>(srcImage, srcImagePrefiltered, point(1.f), BORDER_MIRROR, stdPass)));
        require(upsampleTexCubicBsplineFast(srcImagePrefiltered, dstImageTest, 1.f/resampleFactor, stdPass));
    }

    ////

    if (test == Test::UnprefilterUnserBspline)
    {
        GPU_MATRIX_ALLOC(srcImagePrefiltered, FloatPixel, srcSize);
        require((bsplineCubicPrefilter<FloatPixel, FloatPixel, FloatPixel>(makeConst(srcImage), srcImagePrefiltered, point(1.f), BORDER_MIRROR, stdPass)));

        GPU_MATRIX_ALLOC(tmp, int8, srcSize);
        require((bsplineCubicUnprefilter<FloatPixel, FloatPixel, int8>(makeConst(srcImagePrefiltered), tmp, point(1.f), BORDER_MIRROR, stdPass)));

        require(convertToFloatPixelEx(tmp, dstImageTest, 1.f, stdPass));
    }

    ////

    if (displayType == DisplayDestination)
    {
        int version = kit.display.circularIndex(2);

        require(kit.gpuImageConsole.addMatrixEx(version ? dstImageTest : dstImage,
            kit.display.factor * pixelMin(), kit.display.factor * pixelMax(), point(1.f),
            INTERP_NEAREST, dstSize, BORDER_ZERO,
            paramMsg(STR("Destination Image (%)"), version ? STR("Test") : STR("Ref")), stdPass));
    }

    //----------------------------------------------------------------
    //
    // Check result
    //
    //----------------------------------------------------------------

    GPU_MATRIX_ALLOC(error, float32, dstSize);
    require(computeError(dstImage, dstImageTest, error, stdPass));

    ////

    if (displayType == DisplayError)
    {
        require(kit.gpuImageConsole.addMatrixEx(error, -kit.display.factor, +kit.display.factor, point(1.f),
            INTERP_NEAREST, dstSize, BORDER_ZERO, STR("Error"), stdPass));
    }

    ////

    MATRIX_ALLOC_FOR_GPU_EXCH(errorCpuFull, float32, dstSize);
    GpuCopyThunk gpuCopy;
    require(gpuCopy(error, errorCpuFull, stdPass));
    gpuCopy.waitClear();

    ////

    Matrix<const float32> errorCpu = errorCpuFull;

    if (allv(errorSpatialMargin > 0 && 2*errorSpatialMargin <= dstSize))
        REQUIRE(errorCpuFull.subs(point(errorSpatialMargin()), dstSize - 2*point(errorSpatialMargin()), errorCpu));

    ////

    float32 maxErr = 0;
    require(computeMaxAbsError(errorCpu, maxErr, stdPass));

    float32 meanErr = 0;
    require(computeMeanAbsError(errorCpu, meanErr, stdPass));

    printMsgL(kit, STR("Max err %0 bits, Mean err %1 bits"), fltf(-fastLog2(maxErr), 1), fltf(-fastLog2(meanErr), 1));

    ////

    returnTrue;
}

//----------------------------------------------------------------

#endif

//----------------------------------------------------------------

}
