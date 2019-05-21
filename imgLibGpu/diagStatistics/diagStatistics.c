#include "diagStatistics.h"

#include <cmath>

#include "numbers/float/floatType.h"

//================================================================
//
// computeMeanSquareError
//
//================================================================

stdbool computeMeanSquareError(const Matrix<const float32>& error, float32& meanSquareError, stdPars(CpuFuncKit))
{
    stdBegin;

    if_not (kit.dataProcessing)
        returnTrue;

    MATRIX_EXPOSE(error);

    ////

    float64 globSumErr2 = 0;

    #pragma omp parallel for

    for (Space Y = 0; Y < errorSizeY; ++Y)
    {
        float32 locSumErr2 = 0;

        for (Space X = 0; X < errorSizeX; ++X)
        {
            float32 value = MATRIX_ELEMENT(error, X, Y);
            locSumErr2 += square(value);
        }

        #pragma omp critical
        {
            globSumErr2 += locSumErr2;
        }

    }

    ////

    float64 avgError = globSumErr2 / areaOf(error);
    meanSquareError = float32(sqrt(avgError));

    ////

    stdEnd;
}

//================================================================
//
// computeMeanAbsError
//
//================================================================

stdbool computeMeanAbsError(const Matrix<const float32>& error, float32& meanError, stdPars(CpuFuncKit))
{
    stdBegin;

    if_not (kit.dataProcessing)
        returnTrue;

    MATRIX_EXPOSE(error);

    ////

    float64 globSumErr = 0;

    #pragma omp parallel for

    for (Space Y = 0; Y < errorSizeY; ++Y)
    {
        float32 locSumErr = 0;

        for (Space X = 0; X < errorSizeX; ++X)
        {
            float32 value = MATRIX_ELEMENT(error, X, Y);
            locSumErr += absv(value);
        }

        #pragma omp critical
        {
            globSumErr += locSumErr;
        }

    }

    ////

    float64 avgError = globSumErr / areaOf(error);
    meanError = float32(avgError);

    ////

    stdEnd;
}

//================================================================
//
// computeMeanAndDeviation
//
//================================================================

stdbool computeMeanAndStdev(const Matrix<const float32>& data, float32& resultAvgValue, float32& resultAvgStdev, stdPars(CpuFuncKit))
{
    stdBegin;

    if_not (kit.dataProcessing)
        returnTrue;

    MATRIX_EXPOSE(data);

    ////

    float64 globSumValue = 0;
    float64 globSumValueSq = 0;

    #pragma omp parallel for

    for (Space Y = 0; Y < dataSizeY; ++Y)
    {
        float32 locSumValue = 0;
        float32 locSumValueSq = 0;

        for (Space X = 0; X < dataSizeX; ++X)
        {
            float32 value = MATRIX_ELEMENT(data, X, Y);
            locSumValue += value;
            locSumValueSq += square(value);
        }

        #pragma omp critical
        {
            globSumValue += locSumValue;
            globSumValueSq += locSumValueSq;
        }

    }

    ////

    float64 divArea = 1. / areaOf(data);

    float64 avgValue = divArea * globSumValue;
    float64 avgValueSq = divArea * globSumValueSq;

    float64 avgStdev = sqrt(clampMin(avgValueSq - square(avgValue), 0.0));

    ////

    resultAvgValue = float32(avgValue);
    resultAvgStdev = float32(avgStdev);

    ////

    stdEnd;
}

//================================================================
//
// computeMaxAbsError
//
//================================================================

stdbool computeMaxAbsError(const Matrix<const float32>& error, float32& maxAbsError, stdPars(CpuFuncKit))
{
    stdBegin;

    if_not (kit.dataProcessing)
        returnTrue;

    MATRIX_EXPOSE(error);

    ////

    float32 globMaxError = 0;

    #pragma omp parallel for

    for (Space Y = 0; Y < errorSizeY; ++Y)
    {
        float32 locMaxError = 0;

        for (Space X = 0; X < errorSizeX; ++X)
        {
            float32 errorValue = MATRIX_ELEMENT(error, X, Y);
            locMaxError = maxv(locMaxError, absv(errorValue));
        }

        #pragma omp critical
        {
            globMaxError = maxv(globMaxError, locMaxError);
        }

    }

    ////

    maxAbsError = globMaxError;

    ////

    stdEnd;
}
