#pragma once

#include "compileTools/compileTools.h"
#include "prepTools/prepFor.h"

//================================================================
//
// COMPILE_LOOP
//
// A compile-time loop tool. Usage:
//
// Consider a loop like this one:
//
// for_count (i, count)
//   doSomething(i, extra);
//
// To change it into compile-time loop, write a code like this:
//
// #define ITERATION(i, extra)
//     doSomething(i, extra)
//
// COMPILE_LOOP(MAX_COUNT, count, ITERATION, extra)
//
// #undef ITERATION
//
// During preprocessing, the loop body code is replicated MAX_COUNT times
// with surrounding checks, skipping iterations which don't satisfy (i < count) condition.
//
// To use compile-time loop, (count) value has to be known at compile-time.
// In addition, a simple integer constant MAX_COUNT should be provided, which is known at preprocessing time.
// If specified MAX_COUNT is less than (count), a compilation error occurs.
// If MAX_COUNT is too high, compilation/codegen time can increase unnecessarily.
//
// Currently, only MAX_COUNT <= 256 is supported, otherwise a compilation error occurs.
//
// Two nested loops are supported, COMPILE_LOOP0 and COMPILE_LOOP1 (in any order).
// COMPILE_LOOP is a synonym of COMPILE_LOOP0.
//
//================================================================

//================================================================
//
// COMPILE_LOOP
//
//================================================================

#define COMPILE_LOOP COMPILE_LOOP0

//================================================================
//
// COMPILE_LOOP0
//
//================================================================

#define COMPILE_LOOP0_ITER_(i, count, macro, extra) \
    if (i < (count)) {macro(i, extra)}

#define COMPILE_LOOP0_ITER(i, pars) \
    COMPILE_LOOP0_ITER_(i, PREP_ARG3_0 pars, PREP_ARG3_1 pars, PREP_ARG3_2 pars)

#define COMPILE_LOOP0(maxCount, count, macro, extra) \
    COMPILE_ASSERT((count) <= (maxCount)); \
    PREP_FOR0(maxCount, COMPILE_LOOP0_ITER, (count, macro, extra))

//================================================================
//
// COMPILE_LOOP1
//
//================================================================

#define COMPILE_LOOP1_ITER_(i, count, macro, extra) \
    if (i < (count)) {macro(i, extra)}

#define COMPILE_LOOP1_ITER(i, pars) \
    COMPILE_LOOP1_ITER_(i, PREP_ARG3_0 pars, PREP_ARG3_1 pars, PREP_ARG3_2 pars)

#define COMPILE_LOOP1(maxCount, count, macro, extra) \
    COMPILE_ASSERT((count) <= (maxCount)); \
    PREP_FOR1(maxCount, COMPILE_LOOP1_ITER, (count, macro, extra))
