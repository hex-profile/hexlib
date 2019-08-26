#pragma once

#include "prepTools/prepBase.h"
#include "prepTools/prepIncDec.h"

//================================================================
//
// PREP_ITER_DEPTH
//
//================================================================

#define PREP_ITER_DEPTH NONE

//================================================================
//
// PREP_ITER_* variables
//
//================================================================

#define PREP_ITER_INDEX \
    PREP_PASTE(PREP_ITER_INDEX_, PREP_ITER_DEPTH)

#define PREP_ITER_MIN \
    PREP_PASTE(PREP_ITER_MIN_, PREP_ITER_DEPTH)

#define PREP_ITER_MAX \
    PREP_PASTE(PREP_ITER_MAX_, PREP_ITER_DEPTH)

#define PREP_ITER_FILE \
    PREP_PASTE(PREP_ITER_FILE_, PREP_ITER_DEPTH)

#define PREP_ITER_PARAM \
    PREP_PASTE(PREP_ITER_PARAM_, PREP_ITER_DEPTH)

//================================================================
//
// PREP_ITERATE()
//
//================================================================

#define PREP_ITERATE() PREP_PASTE(PREP_ITERATE_FROM_DEPTH_, PREP_ITER_DEPTH)

//----------------------------------------------------------------

#define PREP_ITERATE_FROM_DEPTH_NONE \
    "prepTools/prepIterate/prepIterate0.h"

#define PREP_ITERATE_FROM_DEPTH_0 \
    "prepTools/prepIterate/prepIterate1.h"

#define PREP_ITERATE_FROM_DEPTH_1 \
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
