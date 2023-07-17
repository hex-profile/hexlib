#include "sleep.h"

#include <chrono>
#include <thread>

#include "numbers/float/floatType.h"

//================================================================
//
// sleep
//
//================================================================

bool sleep(float32 seconds)
{
    ensure(def(seconds) && seconds >= 0);

    ////

    uint32 milliseconds = 0;
    ensure(convertNearest(seconds * 1e3f, milliseconds));

    ////

    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));

    ////

    return true;
}
