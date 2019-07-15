#if defined(__linux__)

#include "threadManagerLinux.h"

#include <unistd.h>
#include <pthread.h>
#include <stdlib.h>

#include "storage/constructDestruct.h"
#include "errorLog/errorLog.h"
#include "errorLog/debugBreak.h"
#include "numbers/int/intType.h"

//================================================================
//
// DEBUG_ONLY
//
//================================================================

#if defined(_DEBUG)
    #define DEBUG_ONLY(code) code
#else
    #define DEBUG_ONLY(code)
#endif

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Critical section
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

class CriticalSectionLinux : public CriticalSectionInterface
{

public:

    void enter()
    {
        DEBUG_BREAK_CHECK(pthread_mutex_lock(&mutex) == 0);
    }

    void leave()
    {
        DEBUG_BREAK_CHECK(pthread_mutex_unlock(&mutex) == 0);
    }

    bool tryEnter()
    {
        return pthread_mutex_trylock(&mutex) == 0;
    }

public:

    inline CriticalSectionLinux()
    {
        ok = true;

        if_not (pthread_mutex_init(&mutex, nullptr) == 0)
            ok = false;
    }

public:

    inline ~CriticalSectionLinux()
    {
        if (ok)
            DEBUG_BREAK_CHECK(pthread_mutex_destroy(&mutex) == 0);
    }

    bool isOk() const {return ok;}

private:

    bool ok = false;
    pthread_mutex_t mutex;

};

//================================================================
//
// ThreadManagerLinux::createCriticalSection
//
//================================================================

stdbool ThreadManagerLinux::createCriticalSection(CriticalSection& section, stdPars(ThreadToolKit))
{
    section.clear();

    auto& sectionEx = section.data.recast<CriticalSectionLinux>();

    constructDefault(sectionEx);
    section.intrface = &sectionEx;

    REQUIRE(sectionEx.isOk());
    return true;
}

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Event
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// ThreadManagerLinux::createEvent
//
//================================================================

stdbool ThreadManagerLinux::createEvent(bool manualReset, EventOwner& event, stdPars(ThreadToolKit))
{
    event.clear();

    ////

    REQUIRE(false); // not implemented

    ////

    returnTrue;
}

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Thread
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// ThreadManagerLinux::createThread
//
//================================================================

stdbool ThreadManagerLinux::createThread(ThreadFunc* threadFunc, void* threadParams, CpuAddrU stackSize, ThreadControl& threadControl, stdPars(ThreadToolKit))
{
    threadControl.waitAndClear();
    REQUIRE(false); // not impl
    return true;
}

//================================================================
//
// ThreadManagerLinux::getCurrentThread
//
//================================================================

stdbool ThreadManagerLinux::getCurrentThread(ThreadControl& threadControl, stdPars(ThreadToolKit))
{
    threadControl.waitAndClear();
    REQUIRE(false); // not impl
    return true;
}

//----------------------------------------------------------------

#endif
