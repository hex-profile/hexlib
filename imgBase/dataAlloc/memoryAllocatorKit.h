#pragma once

#include "kit/kit.h"
#include "storage/addrSpace.h"

//================================================================
//
// DataProcessingKit
//
// A kit to support allocation estimation (data processing skipping).
//
// If the flag is not set, the application code should
// skip ALL data processing and perform only data allocation.
//
// The application code should be extremely fast in this mode.
//
//================================================================

KIT_CREATE1(DataProcessingKit, bool, dataProcessing);

//================================================================
//
// AllocatorObject
//
//================================================================

template <typename AddrU>
struct AllocatorObject;

//================================================================
//
// CpuFastAllocKit
//
//================================================================

KIT_CREATE1(CpuFastAllocKit, AllocatorObject<CpuAddrU>&, cpuFastAlloc);
