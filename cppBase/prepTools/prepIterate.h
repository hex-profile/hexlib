#pragma once

#include "prepTools/prepBase.h"
#include "prepTools/prepIncDec.h"
#include "prepTools/prepArg.h"

//================================================================
//
// PREP_ITER_DEPTH
//
//================================================================

#define PREP_ITER_DEPTH 0

//================================================================
//
// PREP_ITER_PASTE
//
//================================================================

#define PREP_ITER_PASTE(X, Y) \
    PREP_ITER_PASTE2(X, Y)

#define PREP_ITER_PASTE2(X, Y) \
    X##Y

//================================================================
//
// PREP_ITER_* variables
//
//================================================================

#define PREP_ITER_INDEX \
    PREP_ITER_PASTE(PREP_ITER_INDEX_, PREP_DEC(PREP_ITER_DEPTH))

#define PREP_ITER_MIN \
    PREP_PASS(PREP_ARG3_0 PREP_ITER_PASTE(PREP_ITER_ARGS_, PREP_DEC(PREP_ITER_DEPTH)))

#define PREP_ITER_MAX \
    PREP_PASS(PREP_ARG3_1 PREP_ITER_PASTE(PREP_ITER_ARGS_, PREP_DEC(PREP_ITER_DEPTH)))

#define PREP_ITER_FILE \
    PREP_PASS(PREP_ARG3_2 PREP_ITER_PASTE(PREP_ITER_ARGS_, PREP_DEC(PREP_ITER_DEPTH)))

//================================================================
//
// PREP_ITERATE
//
//================================================================

#define PREP_ITERATE \
    PREP_ITER_PASTE(PREP_ITERATE_FROM_DEPTH_, PREP_ITER_DEPTH)

//----------------------------------------------------------------

#define PREP_ITERATE_FROM_DEPTH_0 \
    "prepTools/prepIterate/prepIterate0.h"

#define PREP_ITERATE_FROM_DEPTH_1 \
    "prepTools/prepIterate/prepIterate1.h"

#define PREP_ITERATE_FROM_DEPTH_2 \
    "prepTools/prepIterate/prepIterate2.h"

//================================================================
//
// Generator
//
//================================================================

/*
l = int(input("Level "))
print('\n')

for i in range(0, 256 + 1):
    s = \
    '#if PREP_ITER_MIN_{l} <= {i} && {i} <= PREP_ITER_MAX_{l}\n' \
    '    #define PREP_ITER_INDEX_{l} {i}\n' \
    '    #include PREP_ITER_FILE_{l}\n' \
    '    #undef PREP_ITER_INDEX_{l}\n' \
    '#endif\n'

    print(s.format(i=i, l=l))
*/

//================================================================
//
// PREP_ITERATE_ND
//
//================================================================

#define PREP_ITERATE_ND "prepTools/prepIterate/prepIterateND.h"
