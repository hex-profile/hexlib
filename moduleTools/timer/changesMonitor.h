#pragma once

#include "timer/timerKit.h"
#include "timer/timer.h"
#include "storage/disposableObject.h"

//================================================================
//
// ChangesMonitor
// ```
//
//================================================================

class ChangesMonitor
{

public:

    void touch()
    {
        touched = true;
    }

    void touch(bool changed)
    {
        if (changed)
            touched = true;
    }

    bool active(float32 duration, const TimerKit& kit)
    {
        auto currentMoment = kit.timer.moment();

        if (touched)
        {
            lastChange = currentMoment;
            touched = false;
        }

        return lastChange && kit.timer.diff(*lastChange, currentMoment) <= duration;
    }

private:

    bool touched = false;
    DisposableObject<TimeMoment> lastChange;

};
