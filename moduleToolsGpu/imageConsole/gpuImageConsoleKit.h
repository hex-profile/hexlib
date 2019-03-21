#pragma once

#include "kit/kit.h"
#include "numbers/float/floatBase.h"

//================================================================
//
// GpuImageConsole
//
//================================================================

struct GpuImageConsole;

KIT_CREATE2(GpuImageConsoleKit, GpuImageConsole&, gpuImageConsole, float32, displayFactor);
