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

KIT_COMBINE4(CpuFuncKit, ErrorLogKit, ProfilerKit, CpuFastAllocKit, DataProcessingKit);
