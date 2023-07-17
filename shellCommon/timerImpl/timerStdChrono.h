#pragma once

#include "timer/timer.h"

namespace timerStdChrono {

//================================================================
//
// TimerStdChrono
//
//================================================================

class TimerStdChrono : public Timer
{

public:

    TimeMoment moment() const;

    float32 diff(const TimeMoment& t1, const TimeMoment& t2) const;

    TimeMoment add(const TimeMoment& baseMoment, float32 difference) const;

    TimeMicroseconds convertToSystemMicroseconds(const TimeMoment& baseMoment) const;

    TimeMicroseconds convertToSteadyMicroseconds(const TimeMoment& baseMoment) const;

};

//----------------------------------------------------------------

}

using timerStdChrono::TimerStdChrono;
