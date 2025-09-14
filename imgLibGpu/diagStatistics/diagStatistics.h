#pragma once

#include "gpuProcessHeader.h"

//================================================================
//
// computeMeanSquareError
// computeMeanAbsError
//
//================================================================

void computeMeanSquareError(const Matrix<const float32>& error, float32& meanSquareError, stdPars(CpuFuncKit));
void computeMeanAbsError(const Matrix<const float32>& error, float32& meanError, stdPars(CpuFuncKit));
void computeMeanAndStdev(const Matrix<const float32>& data, float32& resultAvgValue, float32& resultAvgStdev, stdPars(CpuFuncKit));
void computeMaxAbsError(const Matrix<const float32>& error, float32& maxAbsError, stdPars(CpuFuncKit));
