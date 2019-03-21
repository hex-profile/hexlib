#pragma once

#include "data/space.h"
#include "storage/typeAlignment.h"

//================================================================
//
// cpuBaseByteAlignment
// cpuRowByteAlignment
//
// Default alignments of container memory beginning
// and matrix row beginning.
//
//================================================================

// Approx size of L1 cache row
static const Space cpuBaseByteAlignment = 64;

// Approx size of CPU SIMD word
static const Space cpuRowByteAlignment = 16;

//----------------------------------------------------------------

COMPILE_ASSERT(cpuBaseByteAlignment % maxNaturalAlignment == 0);
COMPILE_ASSERT(cpuRowByteAlignment % maxNaturalAlignment == 0);
