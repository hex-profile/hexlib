#pragma once

#include "prepTools/prepFor.h"
#include "prepTools/prepIncDec.h"
#include "prepTools/prepIf.h"

//================================================================
//
// PREP_ENUM
//
// Generates a comma-separated list of tokens: a, b, c
//
// PREP_ENUMERATE
//
// Generates a comma-separated list of tokens: a, b, c,
// Can be used for zero arguments.
//
//================================================================

#define PREP_ENUM(n, macro, args) \
    PREP_FOR(n, PREP_ENUM__ITER_CALLER, (macro, args))

#define PREP_ENUM__ITER_CALLER(i, params) \
    PREP_ENUM__ITER(i, PREP_ARG2_0 params, PREP_ARG2_1 params)

#define PREP_ENUM__ITER(i, macro, args) \
    PREP_IF_COMMA(i) \
    macro(i, args)

//----------------------------------------------------------------

#define PREP_ENUMERATE(n, macro, args) \
    PREP_FOR(n, PREP_ENUMERATE__ITER_CALLER, (macro, args))

#define PREP_ENUMERATE__ITER_CALLER(i, params) \
    PREP_ENUMERATE__ITER(i, PREP_ARG2_0 params, PREP_ARG2_1 params)

#define PREP_ENUMERATE__ITER(i, macro, args) \
    macro(i, args),

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
    PREP_ENUM(n, PREP_ENUM_INDEXED__ITER, prefix)

#define PREP_ENUM_INDEXED__ITER(i, prefix) \
    PREP_PASTE(prefix, i)

//----------------------------------------------------------------

#define PREP_ENUMERATE_INDEXED(n, prefix) \
    PREP_ENUMERATE(n, PREP_ENUMERATE_INDEXED__ITER, prefix)

#define PREP_ENUMERATE_INDEXED__ITER(i, prefix) \
    PREP_PASTE(prefix, i)

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
    PREP_FOR(n, PREP_ENUM_LR__ITER, (prefix, postfix)) \

#define PREP_ENUM_LR__ITER(i, args) \
    PREP_IF_COMMA(i) \
    PREP_ENUM_LR__TEXT(i, PREP_ARG2_0 args, PREP_ARG2_1 args)

#define PREP_ENUM_LR__TEXT(i, prefix, postfix) \
    PREP_PASTE3(prefix, i, postfix)

//----------------------------------------------------------------

#define PREP_ENUM_LR_NOPASTE(n, prefix, postfix) \
    PREP_FOR(n, PREP_ENUM_LR_NOPASTE__ITER, (prefix, postfix)) \

#define PREP_ENUM_LR_NOPASTE__ITER(i, args) \
    PREP_IF_COMMA(i) \
    PREP_ENUM_LR_NOPASTE__TEXT(i, PREP_ARG2_0 args, PREP_ARG2_1 args)

#define PREP_ENUM_LR_NOPASTE__TEXT(i, prefix, postfix) \
    prefix i postfix

//================================================================
//
// PREP_ENUMERATE_LR
//
// Generates a comma-separated list of enumerated tokens:
//
// pre0post, pre1post, pre2post,
//
//================================================================

#define PREP_ENUMERATE_LR(n, prefix, postfix) \
    PREP_FOR(n, PREP_ENUMERATE_LR__ITER, (prefix, postfix)) \

#define PREP_ENUMERATE_LR__ITER(i, args) \
    PREP_ENUMERATE_LR__TEXT(i, PREP_ARG2_0 args, PREP_ARG2_1 args)

#define PREP_ENUMERATE_LR__TEXT(i, prefix, postfix) \
    PREP_PASTE3(prefix, i, postfix),

//----------------------------------------------------------------

#define PREP_ENUMERATE_LR_NOPASTE(n, prefix, postfix) \
    PREP_FOR(n, PREP_ENUMERATE_LR_NOPASTE__ITER, (prefix, postfix)) \

#define PREP_ENUMERATE_LR_NOPASTE__ITER(i, args) \
    PREP_ENUMERATE_LR_NOPASTE__TEXT(i, PREP_ARG2_0 args, PREP_ARG2_1 args)

#define PREP_ENUMERATE_LR_NOPASTE__TEXT(i, prefix, postfix) \
    prefix i postfix,

//================================================================
//
// PREP_ENUM_INDEXED_PAIR
//
// Generates a comma-separated list of enumerated token pairs:
//
// T0 v0, T1 v1, T2 v2
//
//================================================================

#define PREP_ENUM_INDEXED_PAIR(n, prefix1, prefix2) \
    PREP_FOR(n, PREP_ENUM_INDEXED_PAIR__ITER, (prefix1, prefix2)) \

#define PREP_ENUM_INDEXED_PAIR__ITER(i, args) \
    PREP_IF_COMMA(i) \
    PREP_PASTE(PREP_ARG2_0 args, i) PREP_PASTE(PREP_ARG2_1 args, i)

//----------------------------------------------------------------

#define PREP_ENUMERATE_INDEXED_PAIR(n, prefix1, prefix2) \
    PREP_FOR(n, PREP_ENUMERATE_INDEXED_PAIR__ITER, (prefix1, prefix2))

#define PREP_ENUMERATE_INDEXED_PAIR__ITER(i, args) \
    PREP_PASTE(PREP_ARG2_0 args, i) PREP_PASTE(PREP_ARG2_1 args, i),
