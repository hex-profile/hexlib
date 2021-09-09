#pragma once

#include "kit/kit.h"
#include "vectorTypes/vectorBase.h"
#include "data/gpuMatrix.h"

//================================================================
//
// GpuRgbFrameKit
//
//================================================================

KIT_CREATE(GpuRgbFrameKit, GpuMatrix<const uint8_x4>, gpuRgbFrame);
