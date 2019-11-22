#pragma once

#include <stddef.h>

#include "prepTools/prepFor.h"
#include "prepTools/prepBase.h"
#include "charType/charType.h"

//================================================================
//
// ITER_TRACE_ENTER
//
//================================================================

#define ITER_TRACE_LOCATION(n, prefix) \
    prefix CT(": Iteration ") PREP_STRINGIZE(n),

////

#define ITER_TRACE_ENTER(index, maxCount) \
    \
    static const CharType* iterLocations[] = \
        {PREP_FOR(maxCount, ITER_TRACE_LOCATION, TRACE_AUTO_LOCATION)}; \
    \
    stdEnterLocation(size_t(index) < size_t(maxCount) ? iterLocations[index] : TRACE_AUTO_LOCATION);
