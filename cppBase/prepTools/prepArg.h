#pragma once

#include "prepTools/prepBase.h"

//================================================================
//
// PREP_ARG*_*
//
// Preprocessor argument retrieval.
//
// Example, several parameters passed as single macro argument myArg:
// (A, B, C)
//
// PREP_ARG3_0 myArg -- gives A
// PREP_ARG3_1 myArg -- gives B
// PREP_ARG3_2 myArg -- gives C
//
//================================================================

#define PREP_ARG1_0(v0) v0

#define PREP_ARG2_0(v0, v1) v0
#define PREP_ARG2_1(v0, v1) v1

#define PREP_ARG3_0(v0, v1, v2) v0
#define PREP_ARG3_1(v0, v1, v2) v1
#define PREP_ARG3_2(v0, v1, v2) v2

#define PREP_ARG4_0(v0, v1, v2, v3) v0
#define PREP_ARG4_1(v0, v1, v2, v3) v1
#define PREP_ARG4_2(v0, v1, v2, v3) v2
#define PREP_ARG4_3(v0, v1, v2, v3) v3

#define PREP_ARG5_0(v0, v1, v2, v3, v4) v0
#define PREP_ARG5_1(v0, v1, v2, v3, v4) v1
#define PREP_ARG5_2(v0, v1, v2, v3, v4) v2
#define PREP_ARG5_3(v0, v1, v2, v3, v4) v3
#define PREP_ARG5_4(v0, v1, v2, v3, v4) v4

#define PREP_ARG6_0(v0, v1, v2, v3, v4, v5) v0
#define PREP_ARG6_1(v0, v1, v2, v3, v4, v5) v1
#define PREP_ARG6_2(v0, v1, v2, v3, v4, v5) v2
#define PREP_ARG6_3(v0, v1, v2, v3, v4, v5) v3
#define PREP_ARG6_4(v0, v1, v2, v3, v4, v5) v4
#define PREP_ARG6_5(v0, v1, v2, v3, v4, v5) v5

#define PREP_ARG7_0(v0, v1, v2, v3, v4, v5, v6) v0
#define PREP_ARG7_1(v0, v1, v2, v3, v4, v5, v6) v1
#define PREP_ARG7_2(v0, v1, v2, v3, v4, v5, v6) v2
#define PREP_ARG7_3(v0, v1, v2, v3, v4, v5, v6) v3
#define PREP_ARG7_4(v0, v1, v2, v3, v4, v5, v6) v4
#define PREP_ARG7_5(v0, v1, v2, v3, v4, v5, v6) v5
#define PREP_ARG7_6(v0, v1, v2, v3, v4, v5, v6) v6

#define PREP_ARG8_0(v0, v1, v2, v3, v4, v5, v6, v7) v0
#define PREP_ARG8_1(v0, v1, v2, v3, v4, v5, v6, v7) v1
#define PREP_ARG8_2(v0, v1, v2, v3, v4, v5, v6, v7) v2
#define PREP_ARG8_3(v0, v1, v2, v3, v4, v5, v6, v7) v3
#define PREP_ARG8_4(v0, v1, v2, v3, v4, v5, v6, v7) v4
#define PREP_ARG8_5(v0, v1, v2, v3, v4, v5, v6, v7) v5
#define PREP_ARG8_6(v0, v1, v2, v3, v4, v5, v6, v7) v6
#define PREP_ARG8_7(v0, v1, v2, v3, v4, v5, v6, v7) v7
