#pragma once

#include "numbers/int/intBase.h"
#include "stdFunc/traceCallstack.h"
#include "charType/charArray.h"
#include "numbers/float/floatBase.h"

//================================================================
//
// ProfilerMoment
//
//================================================================

using ProfilerMoment = uint64;

//================================================================
//
// ProfilerAccum
//
//================================================================

using ProfilerAccum = uint64;

//================================================================
//
// ProfilerNode
//
//================================================================

struct ProfilerNode
{

    TraceLocation location;
    CharArray userName;

    ProfilerNode* lastChild;
    ProfilerNode* prevBrother;

    uint32 counter;
    ProfilerAccum totalTimeSum;

    uint64 totalElemCount;

    ////

    float32 deviceNodeTime;
    float32 deviceTotalTime;

    float32 deviceNodeOverheadTime;
    float32 deviceTotalOverheadTime;

    ////

    inline void init(TraceLocation location)
    {
        this->location = location;
        userName = 0;

        lastChild = 0;
        prevBrother = 0;

        counter = 0;
        totalTimeSum = 0;

        totalElemCount = 0;

        deviceNodeTime = 0;
        deviceTotalTime = 0;

        deviceNodeOverheadTime = 0;
        deviceTotalOverheadTime = 0;
    }
};

//================================================================
//
// profilerUpdateDeviceTreeTime
//
//================================================================

void profilerUpdateDeviceTreeTime(ProfilerNode& node);
