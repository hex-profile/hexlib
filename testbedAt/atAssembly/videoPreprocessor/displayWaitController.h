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
    virtual bool waitForDisplayTime(stdNullPars) =0;
};

//----------------------------------------------------------------

KIT_CREATE1(DisplayDelayerKit, DisplayDelayer&, displayDelayer);

//================================================================
//
// DisplayWaitController
//
//================================================================

class DisplayWaitController
{

public:

    KIT_COMBINE2(Kit, TimerKit, MsgLogsKit);

public:

    void serialize(const ModuleSerializeKit& kit);

    stdbool waitForDisplayTime(stdPars(Kit));

private:

    BoolSwitch<false> waitActive;
    BoolVarStatic<false> displayTime;

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

    bool waitForDisplayTime(stdNullPars)
        {return impl.waitForDisplayTime(stdPassThru);}

    UseType(DisplayWaitController, Kit);

    DisplayDelayerThunk(DisplayWaitController& impl, const Kit& kit)
        : impl(impl), kit(kit) {}

private:

    DisplayWaitController& impl;
    Kit kit;

};
