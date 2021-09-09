#pragma once

#include "errorLog/errorLogKit.h"
#include "stdFunc/profiler.h"
#include "dataAlloc/memoryAllocatorKit.h"

//================================================================
//
// CpuFuncKit
//
// Typical kit for a CPU image processing library function.
//
//================================================================

using CpuFuncKit = KitCombine<ErrorLogKit, ProfilerKit, CpuFastAllocKit, DataProcessingKit>;
