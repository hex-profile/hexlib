#pragma once

#include "timer/timer.h"

//================================================================
//
// SignalPresenter
//
//================================================================

class SignalPresenter
{

public:

    template <typename Kit>
    inline void setSignal(const Kit& kit)
    {
        signalMomentInitialized = true;
        signalMoment = kit.timer.moment();
    }

    inline void clear()
    {
        signalMomentInitialized = false;
    }

public:

    template <typename Kit>
    inline bool isActive(const Kit& kit)
    {
        TimeMoment currentTime = kit.timer.moment();
        return signalMomentInitialized && kit.timer.diff(signalMoment, currentTime) < presentationPeriod;
    }

public:

    inline void setPresentationPeriod(float32 presentationPeriod)
        {this->presentationPeriod = presentationPeriod;}

public:

    inline SignalPresenter()
    {
        presentationPeriod = 0.5f;
    }

private:

    bool signalMomentInitialized = false;
    TimeMoment signalMoment;
    float32 presentationPeriod;

};
