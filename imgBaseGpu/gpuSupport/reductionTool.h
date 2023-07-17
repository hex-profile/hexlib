#pragma once

#include "gpuSupport/reductionTool/reductionToolClassic.h"
#include "gpuSupport/reductionTool/reductionToolModern.h"

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Convenient tools to make reduction.
//
//----------------------------------------------------------------
//
// The tools support arbitrary number of parameters,
// each parameter is given by pair (Type, name).
//
// All parameters are passed as a preprocessor sequence (a) (b) (c), for example:
// ((int, myInt)) ((float, myFloat).
//
//----------------------------------------------------------------
//
// The classic tool also supports N simultaneous parallel reductions:
//
// * outerSize is the number of reductions.
// * outerMember is the current reduction index.
//
//----------------------------------------------------------------
//
// The usage example:
//
// REDUCTION_MODERN_MAKE
// (
//     myName,
//     threadCount, threadIndex, // The reduction size and the index inside the reduction
//
//     // List of pairs {Type, name}
//     ((float32, sumWeight))
//     ((float32_x2, sumWeightValue))
//     ((float32_x2, sumWeightValueSq)),
//
//     {
//         // Inside the body, each computation checks "active" variable.
//         // It checks "active" multiple times as it is efficient on GPU using predicated instructions.
//         // L and R suffixes mean Lvalue and Rvalue.
//
//         if (active) *sumWeightL += *sumWeightR;
//         if (active) *sumWeightValueL += *sumWeightValueR;
//         if (active) *sumWeightValueSqL += *sumWeightValueSqR;
//     }
// )
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================
