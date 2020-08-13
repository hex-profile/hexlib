#if defined(__linux__)

#include "threadManagerLinux.h"

#include <mutex>

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
        mutex.lock();
    }

    void leave()
    {
        mutex.unlock();
    }

    bool tryEnter()
    {
        return mutex.try_lock();
    }

public:

    inline CriticalSectionLinux()
    {
    }

public:

    inline ~CriticalSectionLinux()
    {
    }

private:

    bool ok = false;
    std::mutex mutex;

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

    returnTrue;
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
    returnTrue;
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
    returnTrue;
}

//----------------------------------------------------------------

#endif
