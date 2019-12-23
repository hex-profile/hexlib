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
// GaborParams
//
//================================================================

struct GaborParams
{
    float32 center; 
    float32 sigma; 
};

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
// circularDistance
//
//================================================================

sysinline float32 circularDistance(float32 A, float32 B) // A, B in [0..1) range 
{
    float32 distance = A - B + 1; // [0, 2)
  
    if (distance >= 1) 
        distance -= 1; // [0, 1)

    if (distance >= 0.5f) 
        distance = 1 - distance; // [0, 1/2)

    return distance;
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
    ((float32, orientCenter)) // in circles
    ((float32, orientSigma)) // in circles
    ((Space, orientCount))
    ((float32, orientOfs)) // in orient samples
    ((GaborParams, g0))
    ((Space, scaleStart))
    ((Space, scaleLevels))
    ((float32, scaleFactor))
)
#if DEVCODE
{
    Point<float32> dstSizef = convertFloat32(vGlobSize);
    Point<float32> dstCenter = 0.5f * dstSizef;
    Point<float32> pos = point(Xs, Ys) - dstCenter;

    Point<float32> freq = pos / dstSizef; /* (-1/2, +1/2) */

    float32 freqLen = sqrtf(square(freq.X) + square(freq.Y));
    Point<float32> freqDir = freq / freqLen;
    if (freqLen == 0) freqDir = point(0.f);

    ////

    float32 sumWeight = 0;
    float32 sumWeightValue = 0;

    ////

    for (Space s = 0; s < scaleLevels; ++s)
    {
        float32 scaleRadius = 0.5f * scaleLevels;
        float32 scaleCenter = 0.5f * scaleLevels;
        float32 scaleWeight = cosShape(((s + 0.5f) - scaleCenter) / scaleRadius);

        float32 scale = powf(scaleFactor, float32(scaleStart + s));
        float32 center = scale * g0.center;
        float32 sigma = scale * g0.sigma;

        Space totalOrientations = 2 * orientCount;

        for (Space k = 0; k < totalOrientations; ++k)
        {
            float32 currentOrient = (k + orientOfs) / totalOrientations;
            float32 dist = circularDistance(orientCenter, currentOrient);

            float32 orientWeight = unitGauss(dist / clampMin(orientSigma, 1e-12f));

            Point<float32> rotatedCenter = complexMul(point(center, 0.f), circleCCW(currentOrient));
            Point<float32> ofs = (freq - rotatedCenter) / sigma;

            sumWeightValue += scaleWeight * orientWeight * unitGauss(ofs.X) * unitGauss(ofs.Y);
            sumWeight += scaleWeight * orientWeight;
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
        for (Space i = 0; i < vGlobSize.X; ++i)
        {
            float32 fX = (i + 0.5f) * divFreqSize.X - 0.5f;
            float32 coeff = fourierBlurGauss(fX - centerFreq.X, sigma);
            sum += coeff * devTex2D(srcSampler, (i + 0.5f) * srcTexstep.X, (Y + 0.5f) * srcTexstep.Y);
            sumCoeff += coeff;
        }
    }
    else
    {
        for (Space i = 0; i < vGlobSize.Y; ++i)
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
    auto response = *src;

    if (allv(vGlobSize == COMPILE_ARRAY_SIZE(conservativeFilterFreqOdd)))
        response = response / conservativeFilterFreqOdd[X] / conservativeFilterFreqOdd[Y];
    else if (allv(vGlobSize == COMPILE_ARRAY_SIZE(conservativeFilterFreqEven)))
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
    auto response = *src;

    if (vGlobSize.X == COMPILE_ARRAY_SIZE(conservativeFilterFreqOdd))
        response = response / conservativeFilterFreqOdd[X];
    else if (vGlobSize.X == COMPILE_ARRAY_SIZE(conservativeFilterFreqEven))
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

    for (Space Y = 0; Y < dstSizeY; ++Y)
    {
        MatrixPtr(float32_x2) dst = MATRIX_POINTER(dst, 0, Y);

        for (Space X = 0; X < dstSizeX; ++X, ++dst)
            maxFreq2 = maxv(maxFreq2, vectorLengthSq(*dst));
    }

    ////

    float32 divMaxFreq = maxFreq2 > 0 ? recipSqrt(maxFreq2) : 0;

    ////

    for (Space Y = 0; Y < dstSizeY; ++Y)
    {
        MatrixPtr(float32_x2) dst = MATRIX_POINTER(dst, 0, Y);

        for (Space X = 0; X < dstSizeX; ++X, ++dst)
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

    FourierFilterBankImpl();
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

    NumericVarStaticEx<float32, Space, 0, 1000000, 0> fourierBlurSigma;
    NumericVarStatic<Space, 2, 16, 7> orientationCount;
    NumericVarStaticEx<float32, Space, 0, 1, 0> orientationOffset;
    StandardSignal generateFilterBank;
    BoolSwitch<false> pyramidFilterCompensation;

    NumericVarStaticEx<float32, Space, 1, 16, 4> displayUpsampleFactor;

    NumericVarStaticEx<float32, int, 0, 128, 0> gaborCenter;
    NumericVarStaticEx<float32, int, 0, 128, 0> gaborSigma;
    NumericVarStaticEx<float32, int, 0, 1024, 0> gaborScaleFactor;
    NumericVar<float32> gaborOrientBlurSigma{0, 128, 0};
    NumericVarStatic<int, 0, 64, 0> gaborScaleStart;
    NumericVarStatic<int, 0, 64, 4> gaborScaleLevels;

    BoolVarStatic<true> displayFreqFilter;
    BoolVarStatic<true> displaySpaceFilter;

};

//================================================================
//
// FourierFilterBankImpl::FourierFilterBankImpl
//
//================================================================

FourierFilterBankImpl::FourierFilterBankImpl()
{
    gaborCenter = 0.2477296157f;
    gaborSigma = 0.06073787279f;

    gaborScaleStart = 0;
    gaborScaleLevels = 5;
    gaborScaleFactor = 1/1.5f;

    fourierBlurSigma = 0;
}

//================================================================
//
// FourierFilterBankImpl::serialize
//
//================================================================

void FourierFilterBankImpl::serialize(const ModuleSerializeKit& kit)
{
    displaySwitch.serialize
    (
        kit, STR("Display"), 
        {STR("<Nothing>"), STR("")}, 
        {STR("Gabor Bank Test"), STR("Alt+1")},
        {STR("Separable Gabor"), STR("Alt+2")}
    );

    filterAreaSize.serialize(kit, STR("Filter Size"));
    fourierAreaSize.serialize(kit, STR("Fourier Size"));
    fourierBlurSigma.serialize(kit, STR("Fourier Blur Sigma"));
    orientationCount.serialize(kit, STR("Orientation Count"));
    orientationOffset.serialize(kit, STR("Orientation Offset"));
    generateFilterBank.serialize(kit, STR("Generate Filter Bank"), STR("Ctrl+B"));
    pyramidFilterCompensation.serialize(kit, STR("Pyramid Filter Compensation"), STR(""));
    displayUpsampleFactor.serialize(kit, STR("Display Upsample Factor"));
  
    {
        CFG_NAMESPACE("Gauss Test");

        gaborCenter.serialize(kit, STR("Center"));
        gaborSigma.serialize(kit, STR("Sigma"));
        gaborScaleStart.serialize(kit, STR("Scale Start"));
        gaborScaleLevels.serialize(kit, STR("Scale Levels"));
        gaborScaleFactor.serialize(kit, STR("Scale Factor"));
        gaborOrientBlurSigma.serialize(kit, STR("Orient Blur Sigma"));
    }

    displayFreqFilter.serialize(kit, STR("Display Freq Filter"));
    displaySpaceFilter.serialize(kit, STR("Display Space Filter"));
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
        if_not (fourierAreaSize() == COMPILE_ARRAY_SIZE(conservativeFilterFreqOdd) || fourierAreaSize() == COMPILE_ARRAY_SIZE(conservativeFilterFreqEven))
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

        for (Space k = 0; k < orientationCount; ++k)
        {
            GPU_MATRIX_ALLOC(filterFreq, float32_x2, fourierSize);

            ////

            require
            (
                makeGaborFreqTest
                (
                    filterFreq, 
                    (k + orientationOffset) / totalOrientations, gaborOrientBlurSigma, 
                    orientationCount, orientationOffset,
                    GaborParams{gaborCenter, gaborSigma}, 
                    gaborScaleStart, gaborScaleLevels, gaborScaleFactor, 
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
                require(kit.gpuImageConsole.addVectorImage(filterFreq, kit.display.factor, point(1.f), INTERP_NEAREST, point(0), BORDER_ZERO, 
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
                require(kit.gpuImageConsole.addVectorImage(filterSpace, 1.f/96 * kit.display.factor, point(displayUpsampleFactor()), INTERP_NEAREST, point(0), 
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

        require(kit.gpuImageConsole.addVectorImage(filterFreqSum, 2.f * kit.display.factor, point(1.f), INTERP_NEAREST, point(0), BORDER_ZERO, 
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
            fprintf(outputFile, "constexpr Space PREP_PASTE(FILTER, OrientCount) = %d;\n", orientationCount());
            fprintf(outputFile, "constexpr Space PREP_PASTE(FILTER, SizeX) = %d;\n", filterSize.X);
            fprintf(outputFile, "constexpr Space PREP_PASTE(FILTER, SizeY) = %d;\n", filterSize.Y);
            fprintf(outputFile, "\n");
            fprintf(outputFile, "constexpr float32 PREP_PASTE(FILTER, Freq) = %+.9ff;\n", gaborCenter());
            fprintf(outputFile, "constexpr float32 PREP_PASTE(FILTER, Sigma) = %+.9ff;\n", gaborSigma());
            fprintf(outputFile, "\n");
        }

        ////

        GPU_MATRIX_ALLOC(filterFreqSum, float32_x2, fourierSize);
        require(gpuMatrixSet(filterFreqSum, zeroOf<float32_x2>(), stdPass));

        ////

        for (Space k = 0; k < orientationCount; ++k)
        {

            GPU_ARRAY_ALLOC(filterFreqX, float32_x2, fourierSize.X);
            GPU_ARRAY_ALLOC(filterFreqY, float32_x2, fourierSize.Y);

            Point<float32> freq = gaborCenter() * circleCCW(float32(k + orientationOffset) / orientationCount / 2);
            require(makeSeparableGaborFreq(filterFreqX, freq.X, gaborSigma, stdPass));
            require(makeSeparableGaborFreq(filterFreqY, freq.Y, gaborSigma, stdPass));

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

            require(kit.gpuImageConsole.addVectorImage(filterFreq, kit.display.factor, point(1.f), INTERP_NEAREST, point(0), BORDER_ZERO, 
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

                require(kit.gpuImageConsole.addVectorImage(filterSpace, 1.f/96 * kit.display.factor, point(displayUpsampleFactor()), INTERP_NEAREST, point(0), 
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
                ARRAY_EXPOSE_PREFIX(filterSpaceCpuX, filterX);
                ARRAY_EXPOSE_PREFIX(filterSpaceCpuY, filterY);

                ////

                if (directGaborComputation)
                {
                    Point<float32> freq = gaborCenter() * circleCCW(float32(k + orientationOffset) / orientationCount / 2);

                    float32 spaceSigma = 1.f / ((2 * pi32) * gaborSigma());

                    for (Space i = 0; i < filterSize.X; ++i)
                    {
                        float32 t = (i + 0.5f) - 0.5f * filterSize.X;
                        float32 shape = gauss1(t / spaceSigma) / spaceSigma;
                        filterXPtr[i] = shape * complexConjugate(circleCcw(freq.X * t));
                    }

                    for (Space i = 0; i < filterSize.Y; ++i)
                    {
                        float32 t = (i + 0.5f) - 0.5f * filterSize.Y;
                        float32 shape = gauss1(t / spaceSigma) / spaceSigma;
                        filterYPtr[i] = shape * complexConjugate(circleCcw(freq.Y * t));
                    }
                }

                ////

                if (k == 0)
                {
                    fprintf(outputFile, "static devConstant float32 PREP_PASTE(FILTER, ShapeX)[%d] = { ", filterSize.X);

                    for (Space i = 0; i < filterSize.X; ++i)
                        fprintf(outputFile, "%+.9ff, ", vectorLength(filterXPtr[i]));

                    fprintf(outputFile, "};\n");

                    ////

                    fprintf(outputFile, "static devConstant float32 PREP_PASTE(FILTER, ShapeY)[%d] = { ", filterSize.Y);

                    for (Space i = 0; i < filterSize.Y; ++i)
                        fprintf(outputFile, "%+.9ff, ", vectorLength(filterYPtr[i]));

                    fprintf(outputFile, "};\n\n");
                }

                ////

                fprintf(outputFile, "static devConstant float32_x2 PREP_PASTE(FILTER, Freq%d) = {%+.9ff, %+.9ff};\n", k, freq.X, freq.Y);
                fprintf(outputFile, "static devConstant float32_x2 PREP_PASTE(FILTER, DataX%d)[%d] = { ", k, filterSize.X);

                for (Space i = 0; i < filterSize.X; ++i)
                    fprintf(outputFile, "{%+.9ff, %+.9ff}, ", filterXPtr[i].x, filterXPtr[i].y);

                fprintf(outputFile, "};\n");

                ////

                fprintf(outputFile, "static devConstant float32_x2 PREP_PASTE(FILTER, DataY%d)[%d] = { ", k, filterSize.Y);

                for (Space i = 0; i < filterSize.Y; ++i)
                    fprintf(outputFile, "{%+.9ff, %+.9ff}, ", filterYPtr[i].x, filterYPtr[i].y);

                fprintf(outputFile, "};\n\n");

            }

        }

        ////

        require(kit.gpuImageConsole.addVectorImage(filterFreqSum, kit.display.factor, point(1.f), INTERP_NEAREST, point(0), BORDER_ZERO, 
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
