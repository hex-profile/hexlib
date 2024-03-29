#if defined(__linux__)

#include "threads.h"

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

class MutexLinux : public MutexInterface
{

public:

    void lock()
    {
        mutex.lock();
    }

    void unlock()
    {
        mutex.unlock();
    }

    bool tryLock()
    {
        return mutex.try_lock();
    }

public:

    inline MutexLinux()
    {
    }

public:

    inline ~MutexLinux()
    {
    }

private:

    std::mutex mutex;

};

//================================================================
//
// mutexCreate
//
//================================================================

stdbool mutexCreate(Mutex& section, stdPars(ThreadToolKit))
{
    section.clear();

    auto& sectionEx = section.data.recast<MutexLinux>();

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
// eventCreate
//
//================================================================

stdbool eventCreate(EventOwner& event, stdPars(ThreadToolKit))
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
// threadCreate
//
//================================================================

stdbool threadCreate(ThreadFunc* threadFunc, void* threadParams, CpuAddrU stackSize, ThreadControl& threadControl, stdPars(ThreadToolKit))
{
    threadControl.waitAndClear();
    REQUIRE(false); // not impl
    returnTrue;
}

//================================================================
//
// threadGetCurrent
//
//================================================================

stdbool threadGetCurrent(ThreadControl& threadControl, stdPars(ThreadToolKit))
{
    threadControl.waitAndClear();
    REQUIRE(false); // not impl
    returnTrue;
}

//----------------------------------------------------------------

#endif
