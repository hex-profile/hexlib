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

    bool isThreadProtected() const override
        {return true;} // because std::chrono is protected and the class has no data fields

    TimeMoment moment() const override;

    float32 diff(const TimeMoment& t1, const TimeMoment& t2) const override;

    TimeMoment add(const TimeMoment& baseMoment, float32 difference) const override;

    TimeMicroseconds convertToSystemMicroseconds(const TimeMoment& baseMoment) const override;

    TimeMicroseconds convertToSteadyMicroseconds(const TimeMoment& baseMoment) const override;

};

//----------------------------------------------------------------

}

using timerStdChrono::TimerStdChrono;
