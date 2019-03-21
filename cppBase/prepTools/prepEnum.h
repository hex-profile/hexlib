#pragma once

#include "prepTools/prepArg.h"
#include "prepTools/prepFor.h"
#include "prepTools/prepIncDec.h"

//================================================================
//
// PREP_ENUM
//
// Generates a comma-separated list of tokens
//
// a, b, c
//
//================================================================

#define PREP_ENUM(n, macro, args) \
    PREP_FOR(PREP_DEC(n), PREP_ENUM_ITER_CALLER, (macro, args)) \
    PREP_ENUM_ITER(PREP_DEC(n), macro, args)

#define PREP_ENUM_ITER_CALLER(n, params) \
    PREP_ENUM_ITER(n, PREP_ARG2_0 params, PREP_ARG2_1 params),

#define PREP_ENUM_ITER(n, macro, args) \
    macro(n, args)

//================================================================
//
// PREP_ENUM_INDEXED
//
// Generates a comma-separated list of enumerated tokens:
//
// val0, val1, val2
//
//================================================================

#define PREP_ENUM_INDEXED(n, prefix) \
    PREP_ENUM(n, PREP__ENUM_INDEXED_ITER, prefix)

#define PREP__ENUM_INDEXED_ITER(n, prefix) \
    PREP_PASTE(prefix, n)

//================================================================
//
// PREP_ENUM_LR
//
// Generates a comma-separated list of enumerated tokens:
//
// pre0post, pre1post, pre2post
//
//================================================================

#define PREP_ENUM_LR(n, prefix, postfix) \
    PREP_FOR(PREP_DEC(n), PREP__ENUM_LR_ITER, (prefix, postfix)) \
    PREP__ENUM_LR_TEXT(PREP_DEC(n), prefix, postfix)

#define PREP__ENUM_LR_ITER(n, args) \
    PREP__ENUM_LR_TEXT(n, PREP_ARG2_0 args, PREP_ARG2_1 args),

#define PREP__ENUM_LR_TEXT(n, prefix, postfix) \
    PREP_PASTE3(prefix, n, postfix)

//================================================================
//
// PREP_ENUM_INDEXED_PAIR
//
// Generates a comma-separated list of enumerated token pairs:
//
// T0 v0, T1 v1, T2 v2
//
//================================================================

#define PREP__ENUM_INDEXED_PAIR_TEXT(n, args) \
    PREP_PASTE(PREP_ARG2_0 args, n) PREP_PASTE(PREP_ARG2_1 args, n)

#define PREP__ENUM_INDEXED_PAIR_ITER(n, args) \
    PREP__ENUM_INDEXED_PAIR_TEXT(n, args),

#define PREP_ENUM_INDEXED_PAIR(n, prefix1, prefix2) \
    PREP_FOR(PREP_DEC(n), PREP__ENUM_INDEXED_PAIR_ITER, (prefix1, prefix2)) \
    PREP__ENUM_INDEXED_PAIR_TEXT(PREP_DEC(n), (prefix1, prefix2))
