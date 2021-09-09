#if defined(_WIN32)

#include "threads.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <process.h>
#include <stdlib.h>

#include "storage/constructDestruct.h"
#include "errorLog/errorLog.h"
#include "errorLog/debugBreak.h"
#include "numbers/int/intType.h"
#include "storage/rememberCleanup.h"

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

class MutexWin32 : public MutexInterface
{

public:

    void lock()
        {EnterCriticalSection(&section);}

    void unlock()
        {LeaveCriticalSection(&section);}

    bool tryLock()
        {return TryEnterCriticalSection(&section) != 0;}

public:

    inline MutexWin32()
    {
        InitializeCriticalSection(&section);
        DEBUG_ONLY(testPtr = malloc(5);)
    }

public:

    inline ~MutexWin32()
    {
        DeleteCriticalSection(&section);
        DEBUG_ONLY(free(testPtr); testPtr = 0;)
    }

private:

    CRITICAL_SECTION section;

#if defined(_DEBUG)
    void* testPtr = 0;
#endif

};

//================================================================
//
// mutexCreate
//
//================================================================

stdbool mutexCreate(Mutex& section, stdPars(ThreadToolKit))
{
    section.clear();

    auto& sectionEx = section.data.recast<MutexWin32>();

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

class EventWin32 : public EventInterface
{

public:

    void set()
    {
        DEBUG_BREAK_CHECK(handle != 0);
        DEBUG_BREAK_CHECK(SetEvent(handle) != 0);
    }

    virtual void reset()
    {
        DEBUG_BREAK_CHECK(handle != 0);
        DEBUG_BREAK_CHECK(ResetEvent(handle) != 0);
    }

    void wait()
    {
        DEBUG_BREAK_CHECK(handle != 0);
        DWORD result = WaitForSingleObject(handle, INFINITE);
        DEBUG_BREAK_CHECK(result == WAIT_OBJECT_0);
    }

    bool waitWithTimeout(uint32 timeMs)
    {
        DEBUG_BREAK_CHECK(handle != 0);
        DWORD result = WaitForSingleObject(handle, timeMs);
        DEBUG_BREAK_CHECK(result == WAIT_OBJECT_0 || result == WAIT_TIMEOUT);
        return (result == WAIT_OBJECT_0);
    }

public:

    stdbool create(stdPars(ThreadToolKit))
    {
        clear();

        ////

        handle = CreateEvent(NULL, false, false, NULL);
        REQUIRE(handle != 0);

        ////

        DEBUG_ONLY(testPtr = malloc(3);)

        returnTrue;
    }

public:

    ~EventWin32() {clear();}

public:

    inline void clear()
    {
        if (handle)
        {
            DEBUG_BREAK_CHECK(CloseHandle(handle) != 0);
            handle = 0;
        }

        DEBUG_ONLY(free(testPtr); testPtr = 0;)
    }

private:

    HANDLE handle = 0;

#if defined(_DEBUG)
    void* testPtr = 0;
#endif

};

//================================================================
//
// eventCreate
//
//================================================================

stdbool eventCreate(EventOwner& event, stdPars(ThreadToolKit))
{
    event.clear();

    ////

    auto& eventEx = event.data.recast<EventWin32>();

    constructDefault(eventEx);
    require(eventEx.create(stdPass));

    event.intrface = &eventEx;

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
// ThreadCallerParams
//
//================================================================

struct ThreadCallerParams
{
    ThreadFunc* threadFunc;
    void* threadParams;
};

//================================================================
//
// threadCallerFunc
//
//================================================================

static unsigned __stdcall threadCallerFunc(void* params)
{
    ThreadCallerParams* p = (ThreadCallerParams*) params;
    p->threadFunc(p->threadParams);
    return 0;
}

//================================================================
//
// threadPriorityToWin32
//
//================================================================

int32 threadPriorityToWin32(ThreadPriority priority)
{
    int32 result = THREAD_PRIORITY_NORMAL;

    if (priority == ThreadPriorityMinusMax)
        result = THREAD_PRIORITY_IDLE;

    if (priority == ThreadPriorityMinus2)
        result = THREAD_PRIORITY_LOWEST;

    if (priority == ThreadPriorityMinus1)
        result = THREAD_PRIORITY_BELOW_NORMAL;

    if (priority == ThreadPriorityNormal)
        result = THREAD_PRIORITY_NORMAL;

    if (priority == ThreadPriorityPlus1)
        result = THREAD_PRIORITY_ABOVE_NORMAL;

    if (priority == ThreadPriorityPlus2)
        result = THREAD_PRIORITY_HIGHEST;

    if (priority == ThreadPriorityPlusMax)
        result = THREAD_PRIORITY_TIME_CRITICAL;

    return result;
}

//================================================================
//
// ThreadControllerWin32
//
//================================================================

class ThreadControllerWin32 : public ThreadControlInterface
{

public:

    virtual bool setPriority(ThreadPriority priority)
    {
        ensure(opened);
        ensure(SetThreadPriority(handle, threadPriorityToWin32(priority)) != 0);
        return true;
    }

public:

    ~ThreadControllerWin32() {close();}

public:

    // Open is atomic
    stdbool open(ThreadFunc* threadFunc, void* threadParams, CpuAddrU stackSize, stdPars(ErrorLogKit))
    {
        REQUIRE(!opened);

        ////

        callerParams.threadFunc = threadFunc;
        callerParams.threadParams = threadParams;

        unsigned vcStackSize = 0;
        REQUIRE(convertExact(stackSize, vcStackSize));
        handle = (void*) _beginthreadex(NULL, vcStackSize, threadCallerFunc, &callerParams, 0, NULL);
        REQUIRE(handle != 0);

        ////

        DEBUG_ONLY(testPtr = malloc(11);) // do not check the debug allocation
        opened = true;

        returnTrue;
    }

    void close()
    {
        if (opened) 
        {
            DEBUG_BREAK_CHECK(WaitForSingleObject(handle, INFINITE) == WAIT_OBJECT_0);
            DEBUG_BREAK_CHECK(CloseHandle(handle) != 0);
            DEBUG_ONLY(free(testPtr); testPtr = 0;)
            opened = false;
        }
    }

private:

    bool opened = false;

    HANDLE handle = 0;

    ThreadCallerParams callerParams;

#if defined(_DEBUG)
    void* testPtr = 0;
#endif

};

//================================================================
//
// CurrentThreadWin32
//
//================================================================

class CurrentThreadWin32 : public ThreadControlInterface
{

public:

    virtual bool setPriority(ThreadPriority priority)
    {
        return SetThreadPriority(GetCurrentThread(), threadPriorityToWin32(priority)) != 0;
    }

};

//================================================================
//
// threadCreate
//
//================================================================

stdbool threadCreate(ThreadFunc* threadFunc, void* threadParams, CpuAddrU stackSize, ThreadControl& threadControl, stdPars(ThreadToolKit))
{
    threadControl.waitAndClear();

    auto& threadControlEx = threadControl.data.recast<ThreadControllerWin32>();

    constructDefault(threadControlEx);
    REMEMBER_CLEANUP_EX(destructControl, destruct(threadControlEx));

    require(threadControlEx.open(threadFunc, threadParams, stackSize, stdPassThru));

    ////

    destructControl.cancel();
    threadControl.intrface = &threadControlEx;

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

    auto& threadControlEx = threadControl.data.recast<CurrentThreadWin32>();

    constructDefault(threadControlEx);
    threadControl.intrface = &threadControlEx;

    returnTrue;
}

//----------------------------------------------------------------

#endif
