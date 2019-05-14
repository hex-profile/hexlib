#pragma once

#include <cstddef>

#include "prepTools/prepFor.h"
#include "prepTools/prepBase.h"
#include "charType/charType.h"

//================================================================
//
// PYRAMID_TRACE_ENTER
//
//================================================================

#define PYRAMID_TRACE_MAX_LEVELS 32

////

#define PYRAMID_TRACE_LOCATION(n, prefix) \
    prefix CT(": Level ") PREP_STRINGIZE(n),

////

#define PYRAMID_TRACE_ENTER(level) \
    \
    static const CharType* pyramidLocations[] = \
        {PREP_FOR(PYRAMID_TRACE_MAX_LEVELS, PYRAMID_TRACE_LOCATION, TRACE_AUTO_LOCATION)}; \
    \
    stdEnterLocation(size_t(level) < size_t(PYRAMID_TRACE_MAX_LEVELS) ? pyramidLocations[level] : TRACE_AUTO_LOCATION);
