#pragma once

#include "stdFunc/stdFunc.h"

//================================================================
//
// ProfilerTarget
//
// Abstract interface of profiler target.
//
//================================================================

struct ProfilerTarget
{
    virtual stdbool process(stdPars(ProfilerKit)) =0;
};
