#pragma once

#include "timer/timer.h"
#include "numbers/float/floatType.h"

//================================================================
//
// ReportModerator
//
//================================================================

class ReportModerator
{

public:

    ReportModerator() =default;

    ReportModerator(float32 period)
    {
        setPeriod(period);
    }

    bool setPeriod(float32 period)
    {
        require(def(period) && period > 0);
        reportPeriod = period;
        return true;
    }

    template <typename Kit>
    bool operator () (const Kit& kit)
    {
        TimeMoment currentMoment = kit.timer.moment();

        bool active = false;

        if_not (lastReportDefined && kit.timer.diff(lastReportMoment, currentMoment) <= reportPeriod)
        {
            active = true;
            lastReportDefined = true;
            lastReportMoment = currentMoment;
        }

        return active;
    }

    void reset()
    {
        lastReportDefined = false;
    }

private:

    float32 reportPeriod = 2.f;

    bool lastReportDefined = false;
    TimeMoment lastReportMoment;

};
