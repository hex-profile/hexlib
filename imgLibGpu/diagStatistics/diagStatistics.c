#include "diagStatistics.h"

#include "numbers/float/floatType.h"
#include "numbers/mathIntrinsics.h"

//================================================================
//
// computeMeanSquareError
//
//================================================================

stdbool computeMeanSquareError(const Matrix<const float32>& error, float32& meanSquareError, stdPars(CpuFuncKit))
{
    if_not (kit.dataProcessing)
        returnTrue;

    MATRIX_EXPOSE(error);

    ////

    float64 globSumErr2 = 0;

    pragmaOmp(parallel for)

    for_count (Y, errorSizeY)
    {
        float32 locSumErr2 = 0;

        for_count (X, errorSizeX)
        {
            float32 value = MATRIX_READ(error, X, Y);
            locSumErr2 += square(value);
        }

        pragmaOmp(critical)
        {
            globSumErr2 += locSumErr2;
        }

    }

    ////

    float64 avgError = globSumErr2 / areaOf(error);
    meanSquareError = float32(fastSqrt(avgError));

    ////

    returnTrue;
}

//================================================================
//
// computeMeanAbsError
//
//================================================================

stdbool computeMeanAbsError(const Matrix<const float32>& error, float32& meanError, stdPars(CpuFuncKit))
{
    if_not (kit.dataProcessing)
        returnTrue;

    MATRIX_EXPOSE(error);

    ////

    float64 globSumErr = 0;

    pragmaOmp(parallel for)

    for_count (Y, errorSizeY)
    {
        float32 locSumErr = 0;

        for_count (X, errorSizeX)
        {
            float32 value = MATRIX_ELEMENT(error, X, Y);
            locSumErr += absv(value);
        }

        pragmaOmp(critical)
        {
            globSumErr += locSumErr;
        }

    }

    ////

    float64 avgError = globSumErr / areaOf(error);
    meanError = float32(avgError);

    ////

    returnTrue;
}

//================================================================
//
// computeMeanAndDeviation
//
//================================================================

stdbool computeMeanAndStdev(const Matrix<const float32>& data, float32& resultAvgValue, float32& resultAvgStdev, stdPars(CpuFuncKit))
{
    if_not (kit.dataProcessing)
        returnTrue;

    MATRIX_EXPOSE(data);

    ////

    float64 globSumValue = 0;
    float64 globSumValueSq = 0;

    pragmaOmp(parallel for)

    for_count (Y, dataSizeY)
    {
        float32 locSumValue = 0;
        float32 locSumValueSq = 0;

        for_count (X, dataSizeX)
        {
            float32 value = MATRIX_ELEMENT(data, X, Y);
            locSumValue += value;
            locSumValueSq += square(value);
        }

        pragmaOmp(critical)
        {
            globSumValue += locSumValue;
            globSumValueSq += locSumValueSq;
        }

    }

    ////

    float64 divArea = 1. / areaOf(data);

    float64 avgValue = divArea * globSumValue;
    float64 avgValueSq = divArea * globSumValueSq;

    float64 avgStdev = fastSqrt(clampMin(avgValueSq - square(avgValue), 0.0));

    ////

    resultAvgValue = float32(avgValue);
    resultAvgStdev = float32(avgStdev);

    ////

    returnTrue;
}

//================================================================
//
// computeMaxAbsError
//
//================================================================

stdbool computeMaxAbsError(const Matrix<const float32>& error, float32& maxAbsError, stdPars(CpuFuncKit))
{
    if_not (kit.dataProcessing)
        returnTrue;

    MATRIX_EXPOSE(error);

    ////

    float32 globMaxError = 0;

    pragmaOmp(parallel for)

    for_count (Y, errorSizeY)
    {
        float32 locMaxError = 0;

        for_count (X, errorSizeX)
        {
            float32 errorValue = MATRIX_ELEMENT(error, X, Y);
            locMaxError = maxv(locMaxError, absv(errorValue));
        }

        pragmaOmp(critical)
        {
            globMaxError = maxv(globMaxError, locMaxError);
        }

    }

    ////

    maxAbsError = globMaxError;

    ////

    returnTrue;
}
