#pragma once

#include "prepTools/prepFor.h"
#include "prepTools/prepIncDec.h"
#include "prepTools/prepIf.h"

//================================================================
//
// PREP_NUMERATE
//
// Generates a list of enumerated tokens:
//
// val0 val1 val2
//
//================================================================

#define PREP_NUMERATE(n, prefix) \
    PREP_NUMERATE(n, PREP_NUMERATE__ITER, prefix)

#define PREP_NUMERATE__ITER(i, prefix) \
    PREP_PASTE(prefix, i)

//================================================================
//
// PREP_NUMERATE_LR
//
// Generates a list of enumerated tokens:
//
// pre0post pre1post pre2post
//
//================================================================

#define PREP_NUMERATE_LR(n, prefix, postfix) \
    PREP_FOR(n, PREP_NUMERATE_LR__ITER, (prefix, postfix)) \

#define PREP_NUMERATE_LR__ITER(i, args) \
    PREP_NUMERATE_LR__TEXT(i, PREP_ARG2_0 args, PREP_ARG2_1 args)

#define PREP_NUMERATE_LR__TEXT(i, prefix, postfix) \
    PREP_PASTE3(prefix, i, postfix)

//----------------------------------------------------------------

#define PREP_NUMERATE_LR_NOPASTE(n, prefix, postfix) \
    PREP_FOR(n, PREP_NUMERATE_LR_NOPASTE__ITER, (prefix, postfix)) \

#define PREP_NUMERATE_LR_NOPASTE__ITER(i, args) \
    PREP_NUMERATE_LR_NOPASTE__TEXT(i, PREP_ARG2_0 args, PREP_ARG2_1 args)

#define PREP_NUMERATE_LR_NOPASTE__TEXT(i, prefix, postfix) \
    prefix i postfix
