#pragma once

#include "gpuProcessHeader.h"

//================================================================
//
// computeMeanSquareError
// computeMeanAbsError
//
//================================================================

stdbool computeMeanSquareError(const Matrix<const float32>& error, float32& meanSquareError, stdPars(CpuFuncKit));
stdbool computeMeanAbsError(const Matrix<const float32>& error, float32& meanError, stdPars(CpuFuncKit));
stdbool computeMeanAndStdev(const Matrix<const float32>& data, float32& resultAvgValue, float32& resultAvgStdev, stdPars(CpuFuncKit));
stdbool computeMaxAbsError(const Matrix<const float32>& error, float32& maxAbsError, stdPars(CpuFuncKit));
