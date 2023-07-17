#pragma once

#include "stdFunc/stdFunc.h"
#include "storage/adapters/callable.h"

//================================================================
//
// ProfilerTarget
//
// Abstract interface of profiler target.
//
//================================================================

using ProfilerTarget = Callable<stdbool (stdPars(ProfilerKit))>;
