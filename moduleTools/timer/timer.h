#pragma once

#ifndef HEXLIB_TIMER
#define HEXLIB_TIMER

#include <stdint.h>

#include "storage/opaqueStruct.h"
#include "numbers/float/floatBase.h"

//================================================================
//
// TimeMoment
//
// Opaque structure representing a steady time moment.
//
//================================================================

using TimeMoment = OpaqueStruct<8>;

//================================================================
//
// TimeMicroseconds
//
// System moment: Not steady (!)
//
//================================================================

using TimeMicroseconds = uint64_t;

//================================================================
//
// Timer abstract interface.
//
// The timer is "steady", so that the time continuously
// increases and never jumps back, behaving like a smooth physical time.
//
// As the system time can change abruptly, for example by NTP updates,
// the conversion to system time is not guaranteed to be precise!
//
//================================================================

struct Timer
{
    // Can the instance be shared among multiple threads?
    virtual bool isThreadProtected() const =0;

    // Get the current moment.
    virtual TimeMoment moment() const =0;

    // Get the difference in seconds.
    virtual float32 diff(const TimeMoment& t1, const TimeMoment& t2) const =0;

    // Shift the timer moment by given difference in seconds.
    virtual TimeMoment add(const TimeMoment& baseMoment, float32 difference) const =0;

    // Converts to a system moment in microseconds.
    // May give an error up to a fraction of millisecond.
    // The system time is NOT steady because of NTP updates, etc,
    // so it can't be used for measuring time intervals.
    virtual TimeMicroseconds convertToSystemMicroseconds(const TimeMoment& baseMoment) const =0;

    // Converts to a steady moment in microseconds.
    // It is precise, but the offset of a steady epoch varies every
    // launch of an application.
    virtual TimeMicroseconds convertToSteadyMicroseconds(const TimeMoment& baseMoment) const =0;
};

//----------------------------------------------------------------

#endif
