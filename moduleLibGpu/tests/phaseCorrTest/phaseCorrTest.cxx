#include "phaseCorrTest.h"

#include "gpuDevice/loadstore/storeNorm.h"
#include "gpuSupport/gpuTool.h"
#include "mathFuncs/bsplineShapes.h"
#include "mathFuncs/rotationMath.h"
#include "numbers/float/floatType.h"
#include "readInterpolate/gpuTexCubic.h"
#include "types/lt/ltType.h"
#include "vectorTypes/vectorOperations.h"
#include "vectorTypes/vectorType.h"

#if HOSTCODE
#include "cfgTools/boolSwitch.h"
#include "cfgTools/multiSwitch.h"
#include "dataAlloc/matrixMemory.h"
#include "tests/fourierModel/fourierModel.h"
#include "imageConsole/gpuImageConsole.h"
#include "storage/classThunks.h"
#include "userOutput/printMsgEx.h"
#endif

namespace phaseCorrTest {

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
// gauss1
//
//================================================================

sysinline float32 gauss1(float32 t)
    {return 0.3989422802f * expf(-0.5f * square(t));}

//================================================================
//
// cosShape
//
//================================================================

sysinline float32 cosShape(float32 x, float32 alpha = 0.5f)
{
    x = absv(x);

    float32 t = x * pi32;
    float32 cosVal = cosf(t);
    if (x > 1) cosVal = -1;

    float32 result = alpha + (1 - alpha) * cosVal;

    return saturate(result);
}

//================================================================
//
// extractImageToComplex
//
//================================================================

GPUTOOL_2D
(
    extractImageToComplex,
    ((const MemFloat, src, INTERP_NEAREST, BORDER_ZERO)),
    ((float32_x2, dst)),
    ((Point<float32>, offs)),

    {
        float32 value = tex2DCubic(srcSampler, point(Xs, Ys) + offs, srcTexstep);
        storeNorm(dst, make_float32_x2(value, 0));
    }
)

//================================================================
//
// Dimensions
//
//================================================================

//================================================================
//
// dataWeightFunc
//
//================================================================

#define GAUSS_FIT_FACTOR 5.0f

//----------------------------------------------------------------

sysinline float32 dataWeightFunc(float32 posLen, int dataWeightType)
{
    float32 result = cosShape(2 * posLen); 

    if (dataWeightType)
        result = gauss1(GAUSS_FIT_FACTOR * posLen / 1.5f); 

    return result;
}

//================================================================
//
// freqPreprocessFilter
//
//================================================================

sysinline float32_x2 freqPreprocessFilter(const float32_x2& freq, float32 freqPos)
{
    VECTOR_DECOMPOSE(freq);

    return powf(freqLength, 0.5f) * freqDir;
}

//================================================================
//
// freqWeightFunc
//
//================================================================

sysinline float32 freqWeightFunc(float32 freqLen, int freqWeightType)
{
    float32 result = cosShape(2 * freqLen);

    if (freqWeightType)
        result = 1;

    return result;
}

//================================================================
//
// shapeDataWindow
//
//================================================================

GPUTOOL_2D
(
    shapeDataWindow,
    ((const float32_x2, src, INTERP_NEAREST, BORDER_ZERO)),
    ((float32_x2, dst)),
    ((int, dataWeightType)),

    {
        Point<float32> dstSizef = convertFloat32(vGlobSize);
        LinearTransform<Point<float32>> lt = ltByTwoPoints(point(-1.f), point(-0.5f), dstSizef, point(+0.5f));
        Point<float32> pos = lt(convertFloat32(point(X, Y)));
    
        float32 posLen = sqrtf(square(pos.X) + square(pos.Y));
        float32 coeff = dataWeightFunc(posLen, dataWeightType);
    
        float32_x2 value = devTex2D(srcSampler, Xs * srcTexstep.X, Ys * srcTexstep.Y);
        storeNorm(dst, coeff * value);
    }
)

//----------------------------------------------------------------

GPUTOOL_2D
(
    genDataWeight,
    PREP_EMPTY,
    ((float32, dst)),
    ((int, dataWeightType)),

    {
        Point<float32> dstSizef = convertFloat32(vGlobSize);
        Point<float32> pos = (point(Xs, Ys) / dstSizef) - 0.5f;
        float32 posLen = sqrtf(square(pos.X) + square(pos.Y));
        float32 coeff = dataWeightFunc(posLen, dataWeightType);
        storeNorm(dst, coeff);
    }
)

//================================================================
//
// subtractWeightedAverage
//
//================================================================

#if HOSTCODE

stdbool subtractWeightedAverage(const GpuMatrix<const float32_x2>& src, const GpuMatrix<float32_x2>& dst, int dataWeightType, stdPars(GpuProcessKit))
{
    GpuCopyThunk gpuCopy;

    REQUIRE(equalSize(src, dst));
    Point<Space> size = src.size();

    MATRIX_ALLOC_FOR_GPU_EXCH(img, float32_x2, size);
    require(gpuCopy(src, img, stdPass));
    gpuCopy.waitClear();

    ////

    MATRIX_EXPOSE(img);
    Point<float32> divImgSize = 1.f / convertFloat32(size);

    ////

    if_not (kit.dataProcessing)
        returnTrue;

    //----------------------------------------------------------------
    //
    // Sums
    //
    //----------------------------------------------------------------

    float64 gxSW = 0;
    float64 gxSWV = 0;

    #pragma omp parallel for

    for (Space Y = 0; Y < imgSizeY; ++Y)
    {
        MatrixPtr(float32_x2) ptr = MATRIX_POINTER(img, 0, Y);
        float32 SW = 0;
        float32 SWV = 0;

        for (Space X = 0; X < imgSizeX; ++X, ++ptr)
        {
            Point<float32> pos = (point(X + 0.5f, Y + 0.5f) * divImgSize) - 0.5f;
            float32 posLen = sqrtf(square(pos.X) + square(pos.Y));
            float32 w = dataWeightFunc(posLen, dataWeightType);

            float32 v = ptr->x;

            SW += w;
            SWV += w*v;
        }

        #pragma omp critical
        {
            gxSW += SW;
            gxSWV += SWV;
        }
    }

    //----------------------------------------------------------------
    //
    // 
    //
    //----------------------------------------------------------------
  
    float32 gSW = float32(gxSW);
    float32 gSWV = float32(gxSWV);

    float32 divSW = (gSW > 0) ? 1.f / gSW : 0;
    gSWV *= divSW;

    ////

    float32 avgVal = gSWV;

    //----------------------------------------------------------------
    //
    // Normalize
    //
    //----------------------------------------------------------------

    #pragma omp parallel for

    for (Space Y = 0; Y < imgSizeY; ++Y)
    {
        MatrixPtr(float32_x2) ptr = MATRIX_POINTER(img, 0, Y);

        for (Space X = 0; X < imgSizeX; ++X, ++ptr)
        {
            Point<float32> pos = (point(X + 0.5f, Y + 0.5f) * divImgSize) - 0.5f;
            float32 posLen = sqrtf(square(pos.X) + square(pos.Y));
            float32 w = dataWeightFunc(posLen, dataWeightType);

            ptr->x = w * ((ptr->x) - avgVal);
        }
    }

    //----------------------------------------------------------------
    //
    // 
    //
    //----------------------------------------------------------------

    require(gpuCopy(img, dst, stdPass));
    gpuCopy.waitClear();

    ////

    returnTrue;
}

#endif

//================================================================
//
// conjugateProduct
//
//================================================================

GPUTOOL_2D
(
    conjugateProduct,
    PREP_EMPTY,
    ((const float32_x2, oldSrc))
    ((const float32_x2, newSrc))
    ((float32_x2, dst)),
    PREP_EMPTY,

    {
        float32_x2 oldValue = *oldSrc;
        float32_x2 newValue = *newSrc;

        float32_x2 result = complexMul(newValue, complexConjugate(oldValue));
    
        storeNorm(dst, result);
    }
)

//================================================================
//
// postprocessCorrelation
//
//================================================================

GPUTOOL_2D
(
    postprocessCorrelation,
    ((const float32_x2, src, INTERP_NEAREST, BORDER_ZERO)),
    ((float32, dst)),
    ((Point<Space>, offs))
    ((float32, factor))
    ((float32, empower)),

    {
        float32_x2 value = devTex2D(srcSampler, (Xs + offs.X) * srcTexstep.X, (Ys + offs.Y) * srcTexstep.Y);
        float32 v = clampMin(factor * value.x, 0.f);
        v = powf(v, empower);
        storeNorm(dst, v);
    }
)

//================================================================
//
// findMax
//
//================================================================

#if HOSTCODE

stdbool findMax(const Matrix<const float32>& src, float32& maxValue, stdPars(CpuFuncKit))
{
    if_not (kit.dataProcessing)
        returnTrue;

    ////

    float32 currentMax = -FLT_MAX;
    Point<Space> pos = point(0);

    ////

    MATRIX_EXPOSE(src);

    for (Space Y = 0; Y < srcSizeY; ++Y)
    {
        MatrixPtr(const float32) ptr = MATRIX_POINTER(src, 0, Y);

        for (Space X = 0; X < srcSizeX; ++X, ++ptr)
        {
            float32 value = *ptr;
      
            if (value > currentMax)
            {
                currentMax = value;
                pos = point(X, Y);
            }
        }
    }

    ////

    maxValue = currentMax;

    ////

    returnTrue;
}

#endif

//================================================================
//
// processFreqProd
//
//================================================================

#if HOSTCODE

stdbool processFreqProd
(
    const Matrix<float32_x2>& oldFourier,
    const Matrix<float32_x2>& newFourier,
    const Matrix<float32_x2>& result, 
    bool useMagnitude,
    bool freqWeightType,
    float32& nccCoeff,
    stdPars(ProcessKit)
)
{
    if_not (kit.dataProcessing)
        returnTrue;

    ////

    REQUIRE(equalSize(oldFourier, newFourier, result));
    Point<Space> size = result.size();

    Point<float32> dstSizef = convertFloat32(size);
    Point<float32> dstCenter = 0.5f * dstSizef;

    MATRIX_EXPOSE(oldFourier);
    MATRIX_EXPOSE(newFourier);
    MATRIX_EXPOSE(result);

    //----------------------------------------------------------------
    //
    // Shape freq
    //
    //----------------------------------------------------------------

    #pragma omp parallel for

    for (Space Y = 0; Y < size.Y; ++Y)
    {
        MatrixPtr(float32_x2) oldPtr = MATRIX_POINTER(oldFourier, 0, Y);
        MatrixPtr(float32_x2) newPtr = MATRIX_POINTER(newFourier, 0, Y);

        for (Space X = 0; X < size.X; ++X, ++oldPtr, ++newPtr)
        {
            float32 Xs = X + 0.5f;
            float32 Ys = Y + 0.5f;
            Point<float32> pos = point(Xs, Ys) - dstCenter;
            Point<float32> freq = pos / dstSizef; /* (-1/2, +1/2) */
            float32 freqLen = sqrtf(square(freq.X) + square(freq.Y));
      
            float32 w = freqWeightFunc(freqLen, freqWeightType);

            *oldPtr *= w;
            *newPtr *= w;
        }
    }

    //----------------------------------------------------------------
    //
    // Max harmonic
    //
    //----------------------------------------------------------------

    float32 xMaxHarmonic = 0;
  
    ////

    #pragma omp parallel for

    for (Space Y = 0; Y < size.Y; ++Y)
    {
        MatrixPtr(const float32_x2) oldPtr = MATRIX_POINTER(oldFourier, 0, Y);
        MatrixPtr(const float32_x2) newPtr = MATRIX_POINTER(newFourier, 0, Y);

        float32 maxHarmonic = 0;

        for (Space X = 0; X < size.X; ++X, ++oldPtr, ++newPtr)
        {
            float32 magnitude = minv(vectorLength(*oldPtr), vectorLength(*newPtr));
            maxHarmonic = maxv(maxHarmonic, magnitude);
        }

        #pragma omp critical
        {
            xMaxHarmonic = maxv(xMaxHarmonic, maxHarmonic);
        }
    }

    ////

    float32 divMaxHarmonic = nativeRecipZero(xMaxHarmonic);

    ////

    PRINT_VAR(useMagnitude);

    //----------------------------------------------------------------
    //
    // Preprocess
    //
    //----------------------------------------------------------------

    #pragma omp parallel for

    for (Space Y = 0; Y < size.Y; ++Y)
    {
        MatrixPtr(float32_x2) oldPtr = MATRIX_POINTER(oldFourier, 0, Y);
        MatrixPtr(float32_x2) newPtr = MATRIX_POINTER(newFourier, 0, Y);

        for (Space X = 0; X < size.X; ++X, ++oldPtr, ++newPtr)
        {
            float32_x2 oldVal = *oldPtr;
            float32_x2 newVal = *newPtr;
            VECTOR_DECOMPOSE(oldVal);
            VECTOR_DECOMPOSE(newVal);

            float32 Xs = X + 0.5f;
            float32 Ys = Y + 0.5f;
            Point<float32> pos = point(Xs, Ys) - dstCenter;
            Point<float32> freq = pos / dstSizef; /* (-1/2, +1/2) */
            float32 freqLen = sqrtf(square(freq.X) + square(freq.Y));

            if (useMagnitude)
            {
                oldValLength *= square(freqLen);
                newValLength *= square(freqLen);

                //oldValLength = saturate(oldValLength * divMaxHarmonic * 50.f);
                //newValLength = saturate(newValLength * divMaxHarmonic * 50.f);
            }
            else
            {
                oldValLength = saturate(oldValLength * divMaxHarmonic * 50.f);
                newValLength = saturate(newValLength * divMaxHarmonic * 50.f);
                //oldValLength = 1;
                //newValLength = 1;
            }

            *oldPtr = oldValLength * oldValDir;
            *newPtr = newValLength * newValDir;
        }
    }

    //----------------------------------------------------------------
    //
    // Normalize vectors
    //
    //----------------------------------------------------------------

    float32 xOldSum2 = 0;
    float32 xNewSum2 = 0;
  
    ////

    #pragma omp parallel for

    for (Space Y = 0; Y < size.Y; ++Y)
    {
        MatrixPtr(const float32_x2) oldPtr = MATRIX_POINTER(oldFourier, 0, Y);
        MatrixPtr(const float32_x2) newPtr = MATRIX_POINTER(newFourier, 0, Y);

        float32 oldSum2 = 0;
        float32 newSum2 = 0;

        for (Space X = 0; X < size.X; ++X, ++oldPtr, ++newPtr)
        {
            oldSum2 += vectorLengthSq(*oldPtr);
            newSum2 += vectorLengthSq(*newPtr);
        }

        #pragma omp critical
        {
            xOldSum2 += oldSum2;
            xNewSum2 += newSum2;
        }
    }

    ////

    float32 divOldLen = nativeRecipZero(sqrtf(xOldSum2));
    float32 divNewLen = nativeRecipZero(sqrtf(xNewSum2));

    //----------------------------------------------------------------
    //
    // Apply
    //
    //----------------------------------------------------------------

    float32 xScalarProd = 0;

    ////

    #pragma omp parallel for

    for (Space Y = 0; Y < size.Y; ++Y)
    {
        MatrixPtr(const float32_x2) oldPtr = MATRIX_POINTER(oldFourier, 0, Y);
        MatrixPtr(const float32_x2) newPtr = MATRIX_POINTER(newFourier, 0, Y);
        MatrixPtr(float32_x2) resultPtr = MATRIX_POINTER(result, 0, Y);

        float32 localScalarProd = 0;

        for (Space X = 0; X < size.X; ++X, ++oldPtr, ++newPtr, ++resultPtr)
        {
            float32_x2 prod = divOldLen * divNewLen * complexMul(*newPtr, complexConjugate(*oldPtr));
            *resultPtr = prod;
            localScalarProd += prod.x;
        }

        #pragma omp critical
        {
            xScalarProd += localScalarProd;
        }
    }

    ////

    nccCoeff = xScalarProd;

    ////

    returnTrue;
}

#endif

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// PhaseCorrTest
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// testSize
//
//================================================================

#if HOSTCODE

//================================================================
//
// PhaseCorrTestImpl
//
//================================================================

class PhaseCorrTestImpl
{

public:

    PhaseCorrTestImpl();
    void serialize(const ModuleSerializeKit& kit);
    stdbool process(const Process& o, stdPars(ProcessKit));
    bool isActive() const {return displaySwitch != DisplayNothing;}

private:

    enum DisplayType 
    {
        DisplayNothing, 
        DisplayPhaseCorrelation,
        DisplayCount
    };

    ExclusiveMultiSwitch<DisplayType, DisplayCount, 0xFA775B03> displaySwitch;

    BoolSwitch<false> useMagnitude;
    BoolSwitch<false> dataWeightType;
    BoolSwitch<false> freqWeightType;

    NumericVar<Space> cfgTestSize;
    NumericVar<Space> cfgFreqCount;
    NumericVar<Space> displayUpsampleFactor;
    NumericVar<float32> corrDisplayedHardMax;
    NumericVar<float32> cfgMinPeriod;

};

//================================================================
//
// PhaseCorrTestImpl::PhaseCorrTestImpl
//
//================================================================

PhaseCorrTestImpl::PhaseCorrTestImpl()
    :
    corrDisplayedHardMax(0, 1, 0.125f),
    cfgTestSize(4, 512, 33),
    cfgFreqCount(4, 512, 33),
    displayUpsampleFactor(1, 16, 4),
    cfgMinPeriod(2.f, 32.f, 2.f)
{
}

//================================================================
//
// PhaseCorrTestImpl::serialize
//
//================================================================

void PhaseCorrTestImpl::serialize(const ModuleSerializeKit& kit)
{
    displaySwitch.serialize
    (
        kit, STR("Display"), 
        {STR("<Nothing>"), STR("")}, 
        {STR("Phase Correlation"), STR("")}
    );

    useMagnitude.serialize(kit, STR("Use Magnitude"), STR(""));
    corrDisplayedHardMax.serialize(kit, STR("Corr Displayed Hard Max"));
    dataWeightType.serialize(kit, STR("Data Weight Type"), STR(""));
    freqWeightType.serialize(kit, STR("Freq Weight Type"), STR(""));
    cfgTestSize.serialize(kit, STR("Test Size"));
    cfgFreqCount.serialize(kit, STR("Frequencies Count"));
    displayUpsampleFactor.serialize(kit, STR("Display Upsample Factor"));
    cfgMinPeriod.serialize(kit, STR("Min Period in Pixels"));
}

//================================================================
//
// PhaseCorrTestImpl::process
//
//================================================================

stdbool PhaseCorrTestImpl::process(const Process& o, stdPars(ProcessKit))
{
    DisplayType displayType = kit.verbosity >= Verbosity::On ? displaySwitch : DisplayNothing;

    //
    // sincos table
    //

    CircleTableHolder circleTable;
    require(circleTable.realloc(256, stdPass));

    //----------------------------------------------------------------
    //
    // Phase correlation
    //
    //----------------------------------------------------------------

    Point<Space> phaseDisplayUpsample = point(displayUpsampleFactor());

    ////

    GpuCopyThunk gpuCopy;

    ////

    Point<Space> testSize = point(cfgTestSize());
    Point<Space> freqSize = point(cfgFreqCount());
    const float32 minPeriod = cfgMinPeriod;

    ////

    GPU_MATRIX_ALLOC(oldImage, float32_x2, testSize);

    {
        GPU_MATRIX_ALLOC(oldImageRaw, float32_x2, testSize);
        require(extractImageToComplex(o.oldImage, oldImageRaw, o.userPoint - 0.5f * convertFloat32(testSize), stdPass));
        require(subtractWeightedAverage(oldImageRaw, oldImage, dataWeightType, stdPass));
    }

    ////

    GPU_MATRIX_ALLOC(newImage, float32_x2, testSize);

    {
        GPU_MATRIX_ALLOC(newImageRaw, float32_x2, testSize);
        require(extractImageToComplex(o.newImage, newImageRaw, o.userPoint - 0.5f * convertFloat32(testSize) + o.baseVector, stdPass));
        require(subtractWeightedAverage(newImageRaw, newImage, dataWeightType, stdPass));
    }

    ////

    float32 imgMax = 1/32.f * kit.display.factor;

    require
    (
        kit.gpuImageConsole.addMatrixChan
        (
            !displaySide(kit) ? oldImage : newImage, 0, -imgMax, +imgMax, 
            convertFloat32(phaseDisplayUpsample), INTERP_NONE, oldImage.size() * phaseDisplayUpsample, BORDER_ZERO,
            ImgOutputHint(!displaySide(kit) ? STR("oldImage") : STR("newImage")).setTargetConsole(), 
            stdPass
        )
    );

    ////

    GPU_MATRIX_ALLOC(oldFourier, float32_x2, freqSize);
    require(fourierSeparable(oldImage, oldFourier, point(minPeriod), circleTable(), stdPass));

    GPU_MATRIX_ALLOC(newFourier, float32_x2, freqSize);
    require(fourierSeparable(newImage, newFourier, point(minPeriod), circleTable(), stdPass));

    ////

    //kit.gpuImageConsole.addVectorImage(!displaySide(kit) ? oldFourier : newFourier, kit.display.factor, 
    //  convertFloat32(phaseDisplayUpsample), INTERP_NONE, freqSize * phaseDisplayUpsample, BORDER_ZERO,
    //  ImgOutputHint(!displaySide(kit) ? STR("oldFourier") : STR("newFourier")).setTargetConsole(), stdPass);

    ////

    GPU_MATRIX_ALLOC(phaseProdFreq, float32_x2, freqSize);
    float32 freqNcc = 0;

    {
        MATRIX_ALLOC_FOR_GPU_EXCH(oldFourierCpu, float32_x2, freqSize);
        MATRIX_ALLOC_FOR_GPU_EXCH(newFourierCpu, float32_x2, freqSize);
        MATRIX_ALLOC_FOR_GPU_EXCH(prodCpu, float32_x2, freqSize);

        GpuCopyThunk gpuCopy;
        require(gpuCopy(oldFourier, oldFourierCpu, stdPass));
        require(gpuCopy(newFourier, newFourierCpu, stdPass));
        gpuCopy.waitClear();

        require(processFreqProd(oldFourierCpu, newFourierCpu, prodCpu, useMagnitude, freqWeightType, freqNcc, stdPass));
        require(gpuCopy(prodCpu, phaseProdFreq, stdPass));
        gpuCopy.waitClear();
    }

    printMsgL(kit, STR("Fourier NCC at zero offset = %0"), fltf(freqNcc, 3), msgWarn);

    ////

    //kit.gpuImageConsole.addVectorImage(phaseProdFreq, kit.display.factor, 
    //  convertFloat32(phaseDisplayUpsample), INTERP_NONE, freqSize * phaseDisplayUpsample, BORDER_ZERO,
    //  ImgOutputHint(STR("Phase Product")).setTargetConsole(), stdPass);

    ////

    GPU_MATRIX_ALLOC(phaseProdSpace, float32_x2, testSize);
    require(invFourierSeparable(phaseProdFreq, phaseProdSpace, point(minPeriod), circleTable(), false, stdPass));

    {
        Point<Space> copySize = freqSize + point(1);
    
        float32 correlationMultiplier = 1.f;

        GPU_MATRIX_ALLOC(tmp, float32, copySize);
        require(postprocessCorrelation(phaseProdSpace, tmp, (testSize - copySize)/2, correlationMultiplier, 1.f, stdPass));

        MATRIX_ALLOC_FOR_GPU_EXCH(tmpCpu, float32, copySize);

        {
            GpuCopyThunk gpuCopy;
            require(gpuCopy(tmp, tmpCpu, stdPass));
        }

        float32 maxValue = 0;
        require(findMax(tmpCpu, maxValue, stdPass));
        maxValue = clampMin(maxValue, 0.f);

        printMsgL(kit, STR("PhaseCorrTest: Max Value = %0"), fltf(maxValue, 3));

        float32 displayedMax = (corrDisplayedHardMax() != 0) ? clampMin(maxValue, corrDisplayedHardMax()) : maxValue * 1.1f;

        require
        (
            kit.gpuImageConsole.addMatrixEx
            (
                tmp, 0, +displayedMax, convertFloat32(phaseDisplayUpsample), 
                INTERP_NEAREST, phaseDisplayUpsample*copySize, BORDER_ZERO, 
                ImgOutputHint(STR("Correlation Matrix")).setTargetConsole().setNewLine(), 
                stdPass
            )
        );
    }

    ////

    returnTrue;
}

//================================================================
//
// Thunks
//
//================================================================

CLASSTHUNK_CONSTRUCT_DESTRUCT(PhaseCorrTest)
CLASSTHUNK_VOID1(PhaseCorrTest, serialize, const ModuleSerializeKit&)
CLASSTHUNK_BOOL_STD1(PhaseCorrTest, process, const Process&, ProcessKit)
CLASSTHUNK_BOOL_CONST0(PhaseCorrTest, isActive)

//----------------------------------------------------------------

#endif

}
