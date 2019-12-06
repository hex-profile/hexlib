#pragma once

#include "userOutput/printMsg.h"

//================================================================
//
// memoryUsageReport
//
//================================================================

template <typename Kit>
inline bool memoryUsageReport
(
    const CharArray& name,
    const MemoryUsage& stateUsage,
    const MemoryUsage& tempUsage,
    const ReallocActivity& stateActivity,
    const ReallocActivity& tempActivity,
    stdPars(Kit)
)
{
    return printMsg
    (
        kit.localLog,
        STR("%: GPU %M / %M, CPU %M / %M"),
        name,
        fltf(ldexp(float32(stateUsage.gpuMemSize), -20), 1),
        fltf(ldexp(float32(tempUsage.gpuMemSize), -20), 1),
        fltf(ldexp(float32(stateUsage.cpuMemSize), -20), 1),
        fltf(ldexp(float32(tempUsage.cpuMemSize), -20), 1),

        (stateActivity.sysAllocCount || tempActivity.sysAllocCount) ? msgErr :
        (stateActivity.fastAllocCount) ? msgWarn : msgInfo
    );
}
