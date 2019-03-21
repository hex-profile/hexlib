#pragma once

#include "gpuProcessHeader.h"

//================================================================
//
// computeMeanSquareError
// computeMeanAbsError
//
//================================================================

bool computeMeanSquareError(const Matrix<const float32>& error, float32& meanSquareError, stdPars(CpuFuncKit));
bool computeMeanAbsError(const Matrix<const float32>& error, float32& meanError, stdPars(CpuFuncKit));
bool computeMeanAndStdev(const Matrix<const float32>& data, float32& resultAvgValue, float32& resultAvgStdev, stdPars(CpuFuncKit));
bool computeMaxAbsError(const Matrix<const float32>& error, float32& maxAbsError, stdPars(CpuFuncKit));
