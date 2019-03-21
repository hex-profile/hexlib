#include "timerStdChrono.h"

#include <chrono>

namespace timerStdChrono {

using namespace std;
using namespace chrono;

//================================================================
//
// SteadyClock
// SteadyMoment
// SteadyDuration
//
//================================================================

using SteadyClock = steady_clock;
using SteadyMoment = SteadyClock::time_point;
using SteadyDuration = SteadyClock::duration;

using SystemClock = system_clock;

//================================================================
//
// TimerStdChrono::moment
//
//================================================================

TimeMoment TimerStdChrono::moment() const
{
    TimeMoment result;
    static_assert(sizeof(SteadyMoment) <= sizeof(TimeMoment), "");
    static_assert(alignof(SteadyMoment) <= alignof(TimeMoment), "");
    (SteadyMoment&) result = SteadyClock::now();
    return result;
}

//================================================================
//
// TimerStdChrono::diff
//
//================================================================

float32 TimerStdChrono::diff(const TimeMoment& t1, const TimeMoment& t2) const
{
    const SteadyMoment& m1 = (SteadyMoment&) t1;
    const SteadyMoment& m2 = (SteadyMoment&) t2;

    duration<float32> result = m2 - m1;
    return result.count();
}

//================================================================
//
// TimerStdChrono::add
//
//================================================================

TimeMoment TimerStdChrono::add(const TimeMoment& baseMoment, float32 difference) const
{
    TimeMoment result;

    duration<float32> durationFloat(difference);

    SteadyMoment oldMoment = (const SteadyMoment&) baseMoment;
    SteadyMoment newMoment = oldMoment + duration_cast<SteadyDuration>(durationFloat);

    (SteadyMoment&) result = newMoment;
    return result;
}

//================================================================
//
// convertClock
//
// The idea from:
// https://stackoverflow.com/questions/35282308/convert-between-c11-clocks
//
//================================================================

template <typename SteadyDuration, typename Repr = typename SteadyDuration::rep>
constexpr SteadyDuration maxDuration() noexcept
    {return SteadyDuration{numeric_limits<Repr>::max()};}

//----------------------------------------------------------------

template <typename SteadyDuration>
constexpr SteadyDuration absDuration(const SteadyDuration d) noexcept
    {return SteadyDuration{d.count() < 0 ? -d.count() : d.count()};}

//----------------------------------------------------------------

template
<
    typename DstMoment,
    typename SrcMoment,
    typename DstDuration = typename DstMoment::duration,
    typename SrcDuration = typename SrcMoment::duration,
    typename DstClock = typename DstMoment::clock,
    typename SrcClock = typename SrcMoment::clock
>
DstMoment convertClock(SrcMoment srcMoment, SrcDuration tolerance = std::chrono::nanoseconds{100}, int maxAttempts = 10)
{
    auto srcNow = SrcMoment{};
    auto dstNow = DstMoment{};
    auto bestDelta = maxDuration<SrcDuration>();

    int iterCount = 0;

    for (;;) // at least one attempt so that srcNow and dstNow are initialized for sure (!)
    {
        auto srcBefore = SrcClock::now();
        auto dstBetween = DstClock::now();
        auto srcAfter = SrcClock::now();

        auto srcDiff = srcAfter - srcBefore;
        auto delta = absDuration(srcDiff);

        if (delta < bestDelta)
        {
            srcNow = srcBefore + srcDiff / 2;
            dstNow = dstBetween;
            bestDelta = delta;
        }

        if (++iterCount >= maxAttempts)
            break;

        if (bestDelta <= tolerance)
            break;
    }

    return dstNow + duration_cast<DstDuration>(srcMoment - srcNow);
}

//================================================================
//
// TimerStdChrono::convertToSystemMicroseconds
//
//================================================================

TimeMicroseconds TimerStdChrono::convertToSystemMicroseconds(const TimeMoment& baseMoment) const
{
    SteadyMoment steadyMoment = (const SteadyMoment&) baseMoment;

    auto systemMoment = convertClock<SystemClock::time_point>(steadyMoment);

    auto nowMicroseconds = time_point_cast<microseconds>(systemMoment);
    auto value = nowMicroseconds.time_since_epoch();
    return value.count();
}

//================================================================
//
// TimerStdChrono::convertToSteadyMicroseconds
//
//================================================================

TimeMicroseconds TimerStdChrono::convertToSteadyMicroseconds(const TimeMoment& baseMoment) const
{
    SteadyMoment steadyMoment = (const SteadyMoment&) baseMoment;

    auto nowMicroseconds = time_point_cast<microseconds>(steadyMoment);
    auto value = nowMicroseconds.time_since_epoch();
    return value.count();
}

//----------------------------------------------------------------

}
