#pragma once

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
//         // L and R suffixes mean Lvalue and Rvalue.
//
//         *sumWeightL += *sumWeightR;
//         *sumWeightValueL += *sumWeightValueR;
//         *sumWeightValueSqL += *sumWeightValueSqR;
//     }
// )
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================
