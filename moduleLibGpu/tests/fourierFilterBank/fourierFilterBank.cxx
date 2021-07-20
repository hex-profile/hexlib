#include "fourierFilterBank.h"

#include <stdio.h>

#include "gaussSincResampling/gaussSincResamplingSettings.h"
#include "gpuDevice/loadstore/storeNorm.h"
#include "gpuSupport/gpuTool.h"
#include "mathFuncs/bsplineShapes.h"
#include "mathFuncs/rotationMath.h"
#include "numbers/mathIntrinsics.h"
#include "vectorTypes/vectorOperations.h"
#include "vectorTypes/vectorType.h"

#if HOSTCODE
#include "cfgTools/boolSwitch.h"
#include "cfgTools/multiSwitch.h"
#include "cfgTools/rangeValueControl.h"
#include "dataAlloc/gpuArrayMemory.h"
#include "dataAlloc/gpuMatrixMemory.h"
#include "dataAlloc/matrixMemory.h"
#include "tests/fourierModel/fourierModel.h"
#include "gpuAppliedApi/gpuAppliedApi.h"
#include "gpuMatrixSet/gpuMatrixSet.h"
#include "imageConsole/gpuImageConsole.h"
#include "rndgen/rndgenFloat.h"
#include "storage/classThunks.h"
#include "storage/rememberCleanup.h"
#include "userOutput/printMsgEx.h"
#include "numbers/getBits.h"
#endif

namespace fourierFilterBank {

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Processing functions 
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// unitGauss
//
// Gauss of diameter 1
//
//================================================================

sysinline float32 gauss1(float32 t)
    {return 0.3989422802f * expf(-0.5f * square(t));}

sysinline float32 unitGauss(float32 t)
    {return expf(-0.5f * square(t));}

//================================================================
//
// cosShape
//
//================================================================

sysinline float32 cosShape(float32 x)
{
    x = absv(x);

    float32 t = x * 1.57079632679489662f;
    float32 result = square(cosf(t));

    if (x >= 1)
        result = 0;

    return result;
}

//================================================================
//
// makeGaborFreqTest
//
//================================================================

GPUTOOL_2D_BEG
(
    makeGaborFreqTest,
    PREP_EMPTY,
    ((float32_x2, dst)),
    
    // orient grid
    ((float32, orientCenter)) // in circles
    ((Space, orientCount))
    ((float32, orientOfs)) // in orient samples
    ((float32, orientSigmaStart)) // in orient samples
    ((float32, orientSigmaEnd)) // in orient samples

    // intra-scale params
    ((Space, intraLevels))
    ((float32, gaborCenterStart))
    ((float32, gaborCenterEnd))
    ((float32, gaborSigmaStart))
    ((float32, gaborSigmaEnd))

    // inter-scale params
    ((Space, pyramidStart))
    ((Space, pyramidLevels))
    ((float32, pyramidFactor))
    ((int, pyramidMode))
)
#if DEVCODE
{
    auto dstSizef = convertFloat32(vGlobSize);
    auto originalFreq = point(Xs, Ys) / dstSizef - 0.5f;

    ////

    float32 sumWeight = 0;
    float32 sumWeightValue = 0;

    ////

    for_count (pyramidIndex, pyramidLevels)
    {

        float32 pyramidWeight = 1.f;
        
        if (pyramidMode == 0)
            pyramidWeight = cosShape(((pyramidIndex + 0.5f) - (0.5f * pyramidLevels)) / (0.5f * pyramidLevels));

        if (pyramidMode == 1)
            pyramidWeight = cosShape((pyramidIndex + 0.5f) / pyramidLevels);

        if (pyramidMode == 2)
            pyramidWeight = 1.f;

        ////

        float32 pyramidScale = powf(pyramidFactor, float32(pyramidStart + pyramidIndex));

        for_count (intraIndex, intraLevels)
        {
            float32 intraAlpha = (intraLevels >= 2) ? float32(intraIndex) / (intraLevels - 1) : 0.f;
            float32 intraWeight = 1.f;

            float32 center = pyramidScale * linerp(intraAlpha, gaborCenterStart, gaborCenterEnd);
            float32 sigma = pyramidScale * linerp(intraAlpha, gaborSigmaStart, gaborSigmaEnd);

            float32 orientSigma = linerp(intraAlpha, orientSigmaStart, orientSigmaEnd);

            Space totalOrientations = 2 * orientCount;

            float32 weight = pyramidWeight * intraWeight;

            for_count (k, totalOrientations)
            {
                float32 currentOrient = (k + orientOfs) / totalOrientations;
                float32 dist = circularDistance(orientCenter, currentOrient);

                float32 orientWeight = unitGauss(dist / clampMin(orientSigma, 1e-12f));

                Point<float32> rotatedCenter = complexMul(point(center, 0.f), circleCCW(currentOrient));
                Point<float32> ofs = (originalFreq - rotatedCenter) / sigma;

                float32 gaussValue = unitGauss(ofs.X) * unitGauss(ofs.Y);

                if_not (absv(originalFreq) <= 0.5f * pyramidScale)
                    gaussValue = 0;

                sumWeightValue += weight * orientWeight * gaussValue;
                sumWeight += weight * orientWeight;
            }
        }
    }

    ////

    *dst = make_float32_x2(nativeRecipZero(sumWeight) * sumWeightValue, 0);
}
#endif
GPUTOOL_2D_END

//================================================================
//
// makeSeparableGaborFreq
//
//================================================================

GPUTOOL_2D_BEG
(
    makeSeparableGaborFreq,
    PREP_EMPTY,
    ((float32_x2, dst)),
    ((float32, freqCenter))
    ((float32, freqSigma))
)
#if DEVCODE
{
    float32 dstSizef = convertFloat32(vGlobSize.X);
    float32 dstCenter = 0.5f * dstSizef;
    float32 pos = Xs - dstCenter;
    float32 freq = pos / dstSizef; // (-1/2, +1/2)

    float32 response = unitGauss((freq - freqCenter) / freqSigma);

    *dst = make_float32_x2(response, 0);
}
#endif
GPUTOOL_2D_END

//================================================================
//
// combineSeparableResponses
//
//================================================================

GPUTOOL_2D_BEG
(
    combineSeparableResponses,
    ((float32_x2, freqX, INTERP_NONE, BORDER_ZERO))
    ((float32_x2, freqY, INTERP_NONE, BORDER_ZERO)),
    ((float32_x2, dst)),
    PREP_EMPTY
)
#if DEVCODE
{
    auto fX = devTex2D(freqXSampler, Xs * freqXTexstep.X, 0);
    auto fY = devTex2D(freqYSampler, Ys * freqYTexstep.X, 0);
    *dst = complexMul(fX, fY);
}
#endif
GPUTOOL_2D_END

//================================================================
//
// fourierBlurGauss
//
//================================================================

sysinline float32 fourierBlurGauss(float32 f, float32 sigma)
{
    float32 w = (2 * pi32) * f;
    return expf(-0.5f * square(w) * square(sigma));
}

//================================================================
//
// blurFourierMatrix
//
//================================================================

GPUTOOL_2D_BEG
(
    blurFourierMatrix,
    ((const float32_x2, src, INTERP_NEAREST, BORDER_ZERO)),
    ((float32_x2, dst)),
    ((float32, sigma))
    ((bool, horizontal))
)
#if DEVCODE
{

    auto sum = zeroOf<float32_x2>();
    float32 sumCoeff = 0;

    Point<float32> freqSize = convertFloat32(vGlobSize);
    Point<float32> divFreqSize = 1.f / freqSize; 
    Point<float32> centerFreq = point(Xs, Ys) * divFreqSize - 0.5f;

    if (horizontal)
    {
        for_count (i, vGlobSize.X)
        {
            float32 fX = (i + 0.5f) * divFreqSize.X - 0.5f;
            float32 coeff = fourierBlurGauss(fX - centerFreq.X, sigma);
            sum += coeff * devTex2D(srcSampler, (i + 0.5f) * srcTexstep.X, (Y + 0.5f) * srcTexstep.Y);
            sumCoeff += coeff;
        }
    }
    else
    {
        for_count (i, vGlobSize.Y)
        {
            float32 fY = (i + 0.5f) * divFreqSize.Y - 0.5f;
            float32 coeff = fourierBlurGauss(fY - centerFreq.Y, sigma);
            sum += coeff * devTex2D(srcSampler, (X + 0.5f) * srcTexstep.X, (i + 0.5f) * srcTexstep.Y);
            sumCoeff += coeff;
        }
    }

    ////

    *dst = sumCoeff > 0 ? sum / sumCoeff : zeroOf<float32_x2>();
}
#endif
GPUTOOL_2D_END

//================================================================
//
// compensatePyramidFilter
//
//================================================================

using gaussSincResampling::conservativeFilterFreqOdd;
using gaussSincResampling::conservativeFilterFreqEven;

//----------------------------------------------------------------

GPUTOOL_2D_BEG
(
    compensatePyramidFilter,
    PREP_EMPTY,
    ((const float32_x2, src))
    ((float32_x2, dst)),
    PREP_EMPTY
)
#if DEVCODE
{
    auto response = helpRead(*src);

    if (allv(vGlobSize == COMPILE_ARRAY_SIZE_S(conservativeFilterFreqOdd)))
        response = response / conservativeFilterFreqOdd[X] / conservativeFilterFreqOdd[Y];
    else if (allv(vGlobSize == COMPILE_ARRAY_SIZE_S(conservativeFilterFreqEven)))
        response = response / conservativeFilterFreqEven[X] / conservativeFilterFreqEven[Y];
    else
        response = zeroOf<float32_x2>();

    *dst = response;
}
#endif
GPUTOOL_2D_END

//----------------------------------------------------------------

GPUTOOL_2D_BEG
(
    compensatePyramidFilterSeparable,
    PREP_EMPTY,
    ((const float32_x2, src))
    ((float32_x2, dst)),
    PREP_EMPTY
)
#if DEVCODE
{
    auto response = helpRead(*src);

    if (vGlobSize.X == COMPILE_ARRAY_SIZE_S(conservativeFilterFreqOdd))
        response = response / conservativeFilterFreqOdd[X];
    else if (vGlobSize.X == COMPILE_ARRAY_SIZE_S(conservativeFilterFreqEven))
        response = response / conservativeFilterFreqEven[X];
    else
        response = zeroOf<float32_x2>();

    *dst = response;
}
#endif
GPUTOOL_2D_END

//================================================================
//
// accumulateFreqResponse
//
//================================================================

GPUTOOL_2D
(
    accumulateFreqResponse,
    PREP_EMPTY,
    ((const float32_x2, src))
    ((float32_x2, dst)),
    PREP_EMPTY,
    *dst += *src;
)

//================================================================
//
// normalizeFreqResponse
//
//================================================================

#if HOSTCODE

stdbool normalizeFreqResponse(const Matrix<float32_x2>& dst, stdPars(CpuFuncKit))
{
    if_not (kit.dataProcessing)
        returnTrue;

    MATRIX_EXPOSE(dst);

    ////

    float32 maxFreq2 = 0;

    for_count (Y, dstSizeY)
    {
        auto dst = MATRIX_POINTER(dst, 0, Y);

        for_count_ex (X, dstSizeX, ++dst)
            maxFreq2 = maxv(maxFreq2, vectorLengthSq(helpRead(*dst)));
    }

    ////

    float32 divMaxFreq = maxFreq2 > 0 ? recipSqrt(maxFreq2) : 0;

    ////

    for_count (Y, dstSizeY)
    {
        auto dst = MATRIX_POINTER(dst, 0, Y);

        for_count_ex (X, dstSizeX, ++dst)
            *dst *= divMaxFreq;
    }

    ////

    returnTrue;
}

#endif 

//================================================================
//
// makeGaussFreq
//
//================================================================

GPUTOOL_2D_BEG
(
    makeGaussFreq,
    PREP_EMPTY,
    ((float32_x2, dst)),
    ((float32, freqSigma))
    ((bool, inverse))
    ((float32, maxAmplification))
)
#if DEVCODE
{
    Point<float32> dstSizef = convertFloat32(vGlobSize);
    Point<float32> dstCenter = 0.5f * dstSizef;
    Point<float32> freq = (point(Xs, Ys) - dstCenter) / dstSizef; // (-1/2, +1/2)

    float32 response = unitGauss(vectorLength(freq) / freqSigma);

    if (inverse) 
    {
        response = clampMax(1 / response, maxAmplification);
        if_not (def(response)) response = maxAmplification;
    }

    *dst = make_float32_x2(response, 0);
}
#endif
GPUTOOL_2D_END

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// FourierFilterBankImpl
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

#if HOSTCODE

//================================================================
//
// FourierFilterBankImpl
//
//================================================================

class FourierFilterBankImpl
{

public:

    void serialize(const ModuleSerializeKit& kit);
    bool reallocValid() const {return true;}
    stdbool realloc(stdPars(GpuModuleReallocKit)) {returnTrue;}
    stdbool process(const Process& o, stdPars(GpuModuleProcessKit));
    bool active() const {return displaySwitch != DisplayNothing;}

private:

    enum DisplayType {DisplayNothing, DisplayGaborTest, DisplayGaborSeparable, DisplayCount};
    ExclusiveMultiSwitch<DisplayType, DisplayCount, 0xA54D3281> displaySwitch;

    NumericVarStaticEx<Point<Space>, Space, 1, 512, 22> filterAreaSize;
    NumericVarStaticEx<Point<Space>, Space, 1, 512, 256> fourierAreaSize;

    NumericVar<float32> fourierBlurSigma{0, 1e6f, 0};
    NumericVarStatic<Space, 2, 128, 6> orientationCount;
    NumericVarStaticEx<float32, Space, 0, 1, 0> orientationOffset;
    StandardSignal generateFilterBank;
    BoolSwitch pyramidFilterCompensation{false};

    NumericVarStaticEx<float32, Space, 1, 16, 4> displayUpsampleFactor;

    NumericVar<Point<float32>> gaborOrientSigma{point(0.f), point(16.f), point(0.f)};

    NumericVar<Point<float32>> gaborCenterVar{point(0.f), point(128.f), point(0.25f)}; // in the middle between 0 (brightness invariance) and 1/2 (Nyquist)
    NumericVar<Point<float32>> gaborSigmaCenterFractionVar{point(0.f), point(128.f), point(0.3f)}; // the thickest one to be zero at both 0 and 1/2 with 8 bit accuracy.

    float32 gaborCenter() {return gaborCenterVar().X;}
    float32 gaborSigma() {return gaborSigmaCenterFractionVar().X * gaborCenterVar().X;}

    NumericVar<Space> gaborIntraLevels{1, 128, 1};

    NumericVar<float32> gaborPyramidFactor{1/128.f, 128.f, 1/1.5f};
    NumericVar<int> gaborPyramidStart{0, 64, 0};
    NumericVar<int> gaborPyramidLevels{0, 64, 5};
    NumericVar<int> gaborPyramidMode{0, 2, 0};

    BoolVar displayFreqFilter{true};
    BoolVar displaySpaceFilter{true};

    NumericVar<float32> displayFactor{0, typeMax<float32>(), 1};
    NumericVar<float32> displayFreqFactor{0, typeMax<float32>(), 1};
    NumericVar<float32> displaySpaceFactor{0, typeMax<float32>(), 0.02f};

};

//================================================================
//
// FourierFilterBankImpl::serialize
//
//================================================================

void FourierFilterBankImpl::serialize(const ModuleSerializeKit& kit)
{
    filterAreaSize.serialize(kit, STR("Filter Size"));
    fourierAreaSize.serialize(kit, STR("Fourier Size"));
    fourierBlurSigma.serialize(kit, STR("Fourier Blur Sigma"));
    orientationCount.serialize(kit, STR("Orientation Count"));
    orientationOffset.serialize(kit, STR("Orientation Offset"));
    generateFilterBank.serialize(kit, STR("Generate Filter Bank"), STR("Ctrl+Alt+B"));
    pyramidFilterCompensation.serialize(kit, STR("Pyramid Filter Compensation"), STR(""));
  
    {
        CFG_NAMESPACE("Gauss Test");

        gaborCenterVar.serialize(kit, STR("Center"));
        
        gaborSigmaCenterFractionVar.serialize(kit, STR("Sigma/Center Ratio"));

        gaborIntraLevels.serialize(kit, STR("Intra Levels"));
        
        gaborPyramidStart.serialize(kit, STR("Pyramid Start"));
        gaborPyramidLevels.serialize(kit, STR("Pyramid Levels"));
        gaborPyramidFactor.serialize(kit, STR("Pyramid Factor"));
        gaborPyramidMode.serialize(kit, STR("Pyramid Mode"), STR("0 cos shape, 1 half cos shape, 2 equal weight"));

        gaborOrientSigma.serialize(kit, STR("Orient Sigma"));
    }

    {
        CFG_NAMESPACE("Display");

        displaySwitch.serialize
        (
            kit, STR("Display"), 
            {STR("<Nothing>"), STR("")}, 
            {STR("Gabor Bank Test"), STR("Alt+1")},
            {STR("Separable Gabor"), STR("Alt+2")}
        );

        displayUpsampleFactor.serialize(kit, STR("Display Upsample Factor"));

        displayFreqFilter.serialize(kit, STR("Display Freq Filter"));
        displaySpaceFilter.serialize(kit, STR("Display Space Filter"));

        displayFactor.serialize(kit, STR("Display Factor"));
        displayFreqFactor.serialize(kit, STR("Display Freq Factor"));
        displaySpaceFactor.serialize(kit, STR("Display Space Factor"));
    }
}

//================================================================
//
// FourierFilterBankImpl::process
//
//================================================================

stdbool FourierFilterBankImpl::process(const Process& o, stdPars(GpuModuleProcessKit))
{
    REQUIRE(reallocValid());

    DisplayType displayType = kit.verbosity >= Verbosity::On ? displaySwitch : DisplayNothing;

    if (displayType == DisplayNothing)
        returnTrue;

    ////

    float32 freqFactor = kit.display.factor * displayFactor * displayFreqFactor;
    float32 spaceFactor = kit.display.factor * displayFactor * displaySpaceFactor;

    //----------------------------------------------------------------
    //
    // sincos table
    //
    //----------------------------------------------------------------

    CircleTableHolder circleTable;
    require(circleTable.realloc(256, stdPass));

    //----------------------------------------------------------------
    //
    // Output file
    //
    //----------------------------------------------------------------

    Point<Space> fourierSize = fourierAreaSize;
    Point<Space> filterSize = clampMax(filterAreaSize(), fourierSize);

    const char* fileName = "D:\\testBank.inl";

    FILE* outputFile = 0;
    if (kit.dataProcessing && generateFilterBank) {outputFile = fopen(fileName, "wt"); REQUIRE(outputFile);}
    REMEMBER_CLEANUP1(if (outputFile) fclose(outputFile), FILE*, outputFile);

    ////

    if_not (pyramidFilterCompensation)
        printMsgL(kit, STR("Pyramid filter compensation is disabled"), msgWarn);
    else
    {
        if_not (fourierAreaSize() == COMPILE_ARRAY_SIZE_S(conservativeFilterFreqOdd) || fourierAreaSize() == COMPILE_ARRAY_SIZE_S(conservativeFilterFreqEven))
            printMsgL(kit, STR("Pyramid filter compensation: Fourier size should be %0 or %1"), COMPILE_ARRAY_SIZE(conservativeFilterFreqOdd), COMPILE_ARRAY_SIZE(conservativeFilterFreqEven), msgErr);
    }

    ////

    Space totalOrientations = orientationCount * 2;

    //----------------------------------------------------------------
    //
    // Generate 2D Bank
    //
    //----------------------------------------------------------------

    if (displayType == DisplayGaborTest)
    {

        GPU_MATRIX_ALLOC(filterFreqSum, float32_x2, fourierSize);
        require(gpuMatrixSet(filterFreqSum, zeroOf<float32_x2>(), stdPass));

        ////

        for_count (k, orientationCount())
        {
            GPU_MATRIX_ALLOC(filterFreq, float32_x2, fourierSize);

            ////

            require
            (
                makeGaborFreqTest
                (
                    filterFreq, 

                    // orient grid
                    (k + orientationOffset) / totalOrientations, 
                    orientationCount, orientationOffset,
                    gaborOrientSigma().X, gaborOrientSigma().Y,

                    // intra-scale params
                    gaborIntraLevels,

                    gaborCenterVar().X,
                    gaborCenterVar().Y,

                    gaborSigmaCenterFractionVar().X * gaborCenterVar().X,
                    gaborSigmaCenterFractionVar().Y * gaborCenterVar().Y,

                    // inter-scale params
                    gaborPyramidStart,
                    gaborPyramidLevels,
                    gaborPyramidFactor, 
                    gaborPyramidMode,

                    stdPass
                )
            );

            if (fourierBlurSigma > 0)
            {
                GPU_MATRIX_ALLOC(tmp, float32_x2, fourierSize);
                require(blurFourierMatrix(filterFreq, tmp, filterSize.X * fourierBlurSigma, true, stdPass));
                require(blurFourierMatrix(tmp, filterFreq, filterSize.Y * fourierBlurSigma, false, stdPass));
            }

            if (pyramidFilterCompensation)
                require(compensatePyramidFilter(filterFreq, filterFreq, stdPass));

            if (displayFreqFilter)
            {
                require(kit.gpuImageConsole.addVectorImage(filterFreq, freqFactor, point(1.f), INTERP_NEAREST, point(0), BORDER_ZERO, 
                    ImgOutputHint(STR("Filter(Freq)")).setTargetConsole(), stdPass));
            }

            require(accumulateFreqResponse(filterFreq, filterFreqSum, stdPass));

            ////

            GPU_MATRIX_ALLOC(filterSpaceFull, float32_x2, fourierSize);
            require(invFourierSeparable(filterFreq, filterSpaceFull, point(2.f), circleTable(), true, stdPass));

            ////

            GPU_MATRIX_ALLOC(filterSpace, float32_x2, filterSize);

            {
                GpuCopyThunk gpuCopy;
                GpuMatrix<const float32_x2> tmp;
                REQUIRE(filterSpaceFull.subs((fourierSize - filterSize) / 2, filterSize, tmp));
                require(gpuCopy(tmp, filterSpace, stdPass));
            }

            ////

            if (displaySpaceFilter)
            {
                require(kit.gpuImageConsole.addVectorImage(filterSpace, spaceFactor, point(displayUpsampleFactor()), INTERP_NEAREST, point(0), 
                    BORDER_ZERO, ImgOutputHint(STR("Filter(Space)")).setTargetConsole(), stdPass));
            }

            ////

            MATRIX_ALLOC_FOR_GPU_EXCH(filterFreqCpu, float32_x2, fourierSize);
            MATRIX_ALLOC_FOR_GPU_EXCH(filterSpaceCpu, float32_x2, filterSize);

            {
                GpuCopyThunk gpuCopy;
                require(gpuCopy(filterFreq, filterFreqCpu, stdPass));
                require(gpuCopy(filterSpace, filterSpaceCpu, stdPass));
            }
        }

        ////

        require(kit.gpuImageConsole.addVectorImage(filterFreqSum, freqFactor, point(1.f), INTERP_NEAREST, point(0), BORDER_ZERO, 
            ImgOutputHint(STR("Filter Freq Sum")).setTargetConsole(), stdPass));

    }

    //----------------------------------------------------------------
    //
    // Generate separable Gabor bank
    //
    //----------------------------------------------------------------

    if (displayType == DisplayGaborSeparable)
    {
        bool directGaborComputation = !pyramidFilterCompensation && fourierBlurSigma == 0;

        if (outputFile)
        {
            fprintf(outputFile, "//================================================================\n");
            fprintf(outputFile, "//\n");
            fprintf(outputFile, "// Generated by %s\n", __FILE__);
            fprintf(outputFile, "//\n");
            fprintf(outputFile, "// Pyramid filter compensation is %s.\n", pyramidFilterCompensation ? "ON" : "OFF");
            fprintf(outputFile, "// Direct Gabor computation is %s.\n", directGaborComputation ? "ON" : "OFF");
            fprintf(outputFile, "//\n");
            fprintf(outputFile, "//================================================================\n");
            fprintf(outputFile, "\n");
        }

        if (outputFile)
        {
            fprintf(outputFile, "constexpr Space PREP_PASTE(GABOR_BANK, OrientCount) = %d;\n", orientationCount());
            fprintf(outputFile, "constexpr Space PREP_PASTE(GABOR_BANK, SizeX) = %d;\n", filterSize.X);
            fprintf(outputFile, "constexpr Space PREP_PASTE(GABOR_BANK, SizeY) = %d;\n", filterSize.Y);
            fprintf(outputFile, "\n");
            fprintf(outputFile, "constexpr float32 PREP_PASTE(GABOR_BANK, Freq) = %+.9ff;\n", gaborCenter());
            fprintf(outputFile, "constexpr float32 PREP_PASTE(GABOR_BANK, Sigma) = %+.9ff;\n", gaborSigma());
            fprintf(outputFile, "\n");
        }

        ////

        GPU_MATRIX_ALLOC(filterFreqSum, float32_x2, fourierSize);
        require(gpuMatrixSet(filterFreqSum, zeroOf<float32_x2>(), stdPass));

        ////

        for_count (k, orientationCount())
        {

            GPU_ARRAY_ALLOC(filterFreqX, float32_x2, fourierSize.X);
            GPU_ARRAY_ALLOC(filterFreqY, float32_x2, fourierSize.Y);

            Point<float32> freq = gaborCenter() * circleCCW(float32(k + orientationOffset) / orientationCount / 2);
            require(makeSeparableGaborFreq(filterFreqX, freq.X, gaborSigma(), stdPass));
            require(makeSeparableGaborFreq(filterFreqY, freq.Y, gaborSigma(), stdPass));

            ////

            if (fourierBlurSigma > 0)
            {
                GpuCopyThunk gpuCopy;

                GPU_ARRAY_ALLOC(tmpX, float32_x2, fourierSize.X);
                require(blurFourierMatrix(filterFreqX, tmpX, filterSize.X * fourierBlurSigma, true, stdPass));
                require(gpuCopy(tmpX, filterFreqX, stdPass));

                GPU_ARRAY_ALLOC(tmpY, float32_x2, fourierSize.Y);
                require(blurFourierMatrix(filterFreqY, tmpY, filterSize.Y * fourierBlurSigma, true, stdPass));
                require(gpuCopy(tmpY, filterFreqY, stdPass));
            }

            if (pyramidFilterCompensation)
            {
                require(compensatePyramidFilterSeparable(filterFreqX, filterFreqX, stdPass));
                require(compensatePyramidFilterSeparable(filterFreqY, filterFreqY, stdPass));
            }

            ////

            GPU_MATRIX_ALLOC(filterFreq, float32_x2, fourierSize);
            require(combineSeparableResponses(filterFreqX, filterFreqY, filterFreq, stdPass));

            require(kit.gpuImageConsole.addVectorImage(filterFreq, freqFactor, point(1.f), INTERP_NEAREST, point(0), BORDER_ZERO, 
                ImgOutputHint(STR("Separable(Freq)")).setTargetConsole(), stdPass));

            require(accumulateFreqResponse(filterFreq, filterFreqSum, stdPass));

            ////

            GPU_ARRAY_ALLOC(filterSpaceFullX, float32_x2, fourierSize.X);
            GPU_ARRAY_ALLOC(filterSpaceFullY, float32_x2, fourierSize.Y);

            require(invFourierSeparable(filterFreqX, filterSpaceFullX, point(2.f), circleTable(), true, stdPass));
            require(invFourierSeparable(filterFreqY, filterSpaceFullY, point(2.f), circleTable(), true, stdPass));

            ////

            GPU_MATRIX_ALLOC(filterSpaceFull, float32_x2, fourierSize);
            require(combineSeparableResponses(filterSpaceFullX, filterSpaceFullY, filterSpaceFull, stdPass));

            ////

            GPU_ARRAY_ALLOC(filterSpaceX, float32_x2, filterSize.X);
            GPU_ARRAY_ALLOC(filterSpaceY, float32_x2, filterSize.Y);

            GpuCopyThunk gpuCopy;

            {
                GpuArray<const float32_x2> tmpArr;
        
                REQUIRE(filterSpaceFullX.subs((fourierSize.X - filterSize.X) / 2, filterSize.X, tmpArr));
                require(gpuCopy(tmpArr, filterSpaceX, stdPass));

                REQUIRE(filterSpaceFullY.subs((fourierSize.Y - filterSize.Y) / 2, filterSize.Y, tmpArr));
                require(gpuCopy(tmpArr, filterSpaceY, stdPass));
            }

            ////

            {
                GPU_MATRIX_ALLOC(filterSpace, float32_x2, filterSize);

                require(combineSeparableResponses(filterSpaceX, filterSpaceY, filterSpace, stdPass));

                require(kit.gpuImageConsole.addVectorImage(filterSpace, spaceFactor, point(displayUpsampleFactor()), INTERP_NEAREST, point(0), 
                    BORDER_ZERO, ImgOutputHint(STR("Filter(Space)")).setTargetConsole(), stdPass));
            }

            ////

            ARRAY_ALLOC_FOR_GPU_EXCH(filterSpaceCpuX, float32_x2, filterSize.X);
            require(gpuCopy(filterSpaceX, filterSpaceCpuX, stdPass));

            ARRAY_ALLOC_FOR_GPU_EXCH(filterSpaceCpuY, float32_x2, filterSize.Y);
            require(gpuCopy(filterSpaceY, filterSpaceCpuY, stdPass));

            gpuCopy.waitClear();

            ////

            if (outputFile)
            {
                ARRAY_EXPOSE_EX(filterSpaceCpuX, filterX);
                ARRAY_EXPOSE_EX(filterSpaceCpuY, filterY);

                ////

                if (directGaborComputation)
                {
                    Point<float32> freq = gaborCenter() * circleCCW(float32(k + orientationOffset) / orientationCount / 2);

                    float32 spaceSigma = 1.f / ((2 * pi32) * gaborSigma());

                    auto maxFilterSize = point(256);

                    ////

                    auto maxSum = point(float32{0});

                    for_count (i, maxFilterSize.X)
                    {
                        float32 t = (i + 0.5f) - 0.5f * maxFilterSize.X;
                        maxSum.X += gauss1(t / spaceSigma) / spaceSigma;
                    }

                    for_count (i, maxFilterSize.Y)
                    {
                        float32 t = (i + 0.5f) - 0.5f * maxFilterSize.Y;
                        maxSum.Y += gauss1(t / spaceSigma) / spaceSigma;
                    }

                    ////

                    auto sum = point(float32{0});

                    for_count (i, filterSize.X)
                    {
                        float32 t = (i + 0.5f) - 0.5f * filterSize.X;
                        float32 shape = gauss1(t / spaceSigma) / spaceSigma;
                        filterXPtr[i] = shape * complexConjugate(circleCcw(freq.X * t));
                        sum.X += shape;
                    }

                    ////

                    for_count (i, filterSize.Y)
                    {
                        float32 t = (i + 0.5f) - 0.5f * filterSize.Y;
                        float32 shape = gauss1(t / spaceSigma) / spaceSigma;
                        filterYPtr[i] = shape * complexConjugate(circleCcw(freq.Y * t));
                        sum.Y += shape;
                    }

                    ////

                    auto accBits = getBits(1 - sum / maxSum);
                    printMsgL(kit, STR("Gabor Coverage is {%} bits"), fltf(accBits, 1), msgWarn);
                }

                ////

                if (k == 0)
                {
                    fprintf(outputFile, "static devConstant float32 PREP_PASTE(GABOR_BANK, ShapeX)[%d] = { ", filterSize.X);

                    for_count (i, filterSize.X)
                        fprintf(outputFile, "%+.9ff, ", vectorLength(filterXPtr[i]));

                    fprintf(outputFile, "};\n");

                    ////

                    fprintf(outputFile, "static devConstant float32 PREP_PASTE(GABOR_BANK, ShapeY)[%d] = { ", filterSize.Y);

                    for_count (i, filterSize.Y)
                        fprintf(outputFile, "%+.9ff, ", vectorLength(filterYPtr[i]));

                    fprintf(outputFile, "};\n\n");
                }

                ////

                fprintf(outputFile, "static devConstant float32_x2 PREP_PASTE(GABOR_BANK, Freq%d) = {%+.9ff, %+.9ff};\n", k, freq.X, freq.Y);
                fprintf(outputFile, "static devConstant float32_x2 PREP_PASTE(GABOR_BANK, DataX%d)[%d] = { ", k, filterSize.X);

                for_count (i, filterSize.X)
                    fprintf(outputFile, "{%+.9ff, %+.9ff}, ", filterXPtr[i].x, filterXPtr[i].y);

                fprintf(outputFile, "};\n");

                ////

                fprintf(outputFile, "static devConstant float32_x2 PREP_PASTE(GABOR_BANK, DataY%d)[%d] = { ", k, filterSize.Y);

                for_count (i, filterSize.Y)
                    fprintf(outputFile, "{%+.9ff, %+.9ff}, ", filterYPtr[i].x, filterYPtr[i].y);

                fprintf(outputFile, "};\n\n");

            }

        }

        ////

        require(kit.gpuImageConsole.addVectorImage(filterFreqSum, freqFactor, point(1.f), INTERP_NEAREST, point(0), BORDER_ZERO, 
            ImgOutputHint(STR("Filter Freq Sum")).setTargetConsole(), stdPass));

    }

    //----------------------------------------------------------------
    //
    // Finish
    //
    //----------------------------------------------------------------

    if (outputFile)
        printMsgG(kit, STR("Filter bank is saved to %0."), fileName);

    ////

    returnTrue;
}

//================================================================
//
// Thunks
//
//================================================================

CLASSTHUNK_CONSTRUCT_DESTRUCT(FourierFilterBank)
CLASSTHUNK_VOID1(FourierFilterBank, serialize, const ModuleSerializeKit&)
CLASSTHUNK_BOOL_CONST0(FourierFilterBank, reallocValid)
CLASSTHUNK_BOOL_STD0(FourierFilterBank, realloc, GpuModuleReallocKit)
CLASSTHUNK_BOOL_STD1(FourierFilterBank, process, const Process&, GpuModuleProcessKit)
CLASSTHUNK_BOOL_CONST0(FourierFilterBank, active)

//----------------------------------------------------------------

#endif

}
