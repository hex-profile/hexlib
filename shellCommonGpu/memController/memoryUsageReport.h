#pragma once

#include "userOutput/printMsg.h"
#include "numbers/mathIntrinsics.h"

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
    bool sysAlloc = (stateActivity.sysAllocCount || tempActivity.sysAllocCount);
    bool fastAlloc = stateActivity.fastAllocCount != 0;

    return printMsg
    (
        kit.localLog,
        STR("%: GPU %M : %M, CPU %M : %M%"),
        name,
        fltf(ldexpv(float32(stateUsage.gpuMemSize), -20), 1),
        fltf(ldexpv(float32(tempUsage.gpuMemSize), -20), 1),
        fltf(ldexpv(float32(stateUsage.cpuMemSize), -20), 1),
        fltf(ldexpv(float32(tempUsage.cpuMemSize), -20), 1),

        sysAlloc ? STR(": System Realloc.") : STR(""),

        (sysAlloc || fastAlloc) ? msgWarn : msgInfo
    );
}
