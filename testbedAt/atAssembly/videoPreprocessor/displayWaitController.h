#pragma once

#include "kits/moduleHeader.h"
#include "cfgTools/boolSwitch.h"
#include "timer/timer.h"

//================================================================
//
// DisplayDelayer
//
//================================================================

struct DisplayDelayer
{
    virtual stdbool waitForDisplayTime(stdParsNull) =0;
};

//----------------------------------------------------------------

KIT_CREATE(DisplayDelayerKit, DisplayDelayer&, displayDelayer);

//================================================================
//
// DisplayWaitController
//
//================================================================

class DisplayWaitController
{

public:

    using Kit = KitCombine<TimerKit, MsgLogsKit>;

public:

    void serialize(const ModuleSerializeKit& kit);

    stdbool waitForDisplayTime(stdPars(Kit));

private:

    BoolSwitch waitActive{false};
    BoolVar displayTime{false};

    bool lastOutputInit = false;
    TimeMoment lastOutput;

    NumericVarStaticEx<float32, int32, 0, 10*1000, 0> targetDelayMs;

};

//================================================================
//
// DisplayDelayerThunk
//
//================================================================

class DisplayDelayerThunk : public DisplayDelayer
{

public:

    stdbool waitForDisplayTime(stdParsNull)
        {return impl.waitForDisplayTime(stdPassThru);}

    using Kit = DisplayWaitController::Kit;

    DisplayDelayerThunk(DisplayWaitController& impl, const Kit& kit)
        : impl(impl), kit(kit) {}

private:

    DisplayWaitController& impl;
    Kit kit;

};
