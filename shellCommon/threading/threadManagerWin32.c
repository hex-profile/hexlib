#if defined(_WIN32)

#include "threadManagerWin32.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <process.h>
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

class CriticalSectionWin32 : public CriticalSectionInterface
{

public:

    void enter()
        {EnterCriticalSection(&section);}

    void leave()
        {LeaveCriticalSection(&section);}

    bool tryEnter()
        {return TryEnterCriticalSection(&section) != 0;}

public:

    inline CriticalSectionWin32()
    {
        InitializeCriticalSection(&section);
        DEBUG_ONLY(testPtr = malloc(5);)
    }

public:

    inline ~CriticalSectionWin32()
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
// ThreadManagerWin32::createCriticalSection
//
//================================================================

stdbool ThreadManagerWin32::createCriticalSection(CriticalSection& section, stdPars(ThreadToolKit))
{
    section.clear();

    COMPILE_ASSERT(sizeof(CriticalSectionWin32) <= sizeof(CriticalSectionData));
    CriticalSectionWin32& sectionEx = (CriticalSectionWin32&) section.data;

    constructDefault(sectionEx);
    section.intrface = &sectionEx;

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

class EventWin32 : public DestructibleInterface<Event>
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

    stdbool create(bool manualReset, stdPars(ThreadToolKit))
    {
        stdBegin;

        clear();

        ////

        handle = CreateEvent(NULL, manualReset, false, NULL);
        REQUIRE(handle != 0);

        ////

        DEBUG_ONLY(testPtr = malloc(3);)

        stdEnd;
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
// ThreadManagerWin32::createEvent
//
//================================================================

stdbool ThreadManagerWin32::createEvent(bool manualReset, EventOwner& event, stdPars(ThreadToolKit))
{
    stdBegin;

    event.clear();

    ////

    COMPILE_ASSERT(sizeof(EventWin32) <= sizeof(event.data));
    EventWin32& eventEx = (EventWin32&) event.data;

    constructDefault(eventEx);
    require(eventEx.create(manualReset, stdPass));

    event.intrface = &eventEx;

    ////

    stdEnd;
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
        return SetThreadPriority(handle, threadPriorityToWin32(priority)) != 0;
    }

public:

    ThreadControllerWin32(ThreadFunc* threadFunc, void* threadParams, CpuAddrU stackSize, stdbool& ok, stdPars(ErrorLogKit))
    {
        stdBegin;
        ok = false;

        ////

        callerParams.threadFunc = threadFunc;
        callerParams.threadParams = threadParams;

        unsigned vcStackSize = 0;
        REQUIREV(convertExact(stackSize, vcStackSize));
        handle = (void*) _beginthreadex(NULL, vcStackSize, threadCallerFunc, &callerParams, 0, NULL);
        REQUIREV(handle != 0);

        ////

        DEBUG_ONLY(testPtr = malloc(11);)
        ok = true;
        stdEndv;
    }

public:

    ~ThreadControllerWin32()
    {
        DEBUG_BREAK_CHECK(WaitForSingleObject(handle, INFINITE) == WAIT_OBJECT_0);
        DEBUG_BREAK_CHECK(CloseHandle(handle) != 0);
        DEBUG_ONLY(free(testPtr); testPtr = 0;)
    }

private:

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
// ThreadManagerWin32::createThread
//
//================================================================

stdbool ThreadManagerWin32::createThread(ThreadFunc* threadFunc, void* threadParams, CpuAddrU stackSize, ThreadControl& threadControl, stdPars(ThreadToolKit))
{
    threadControl.waitAndClear();

    COMPILE_ASSERT(sizeof(ThreadControllerWin32) <= sizeof(ThreadControlData));
    ThreadControllerWin32& threadControlEx = (ThreadControllerWin32&) threadControl.data;

    bool ok = false;
    constructParams(threadControlEx, ThreadControllerWin32, (threadFunc, threadParams, stackSize, ok, stdPassThru));
    require(ok);

    threadControl.intrface = &threadControlEx;
    return true;
}

//================================================================
//
// ThreadManagerWin32::getCurrentThread
//
//================================================================

stdbool ThreadManagerWin32::getCurrentThread(ThreadControl& threadControl, stdPars(ThreadToolKit))
{
    threadControl.waitAndClear();

    COMPILE_ASSERT(sizeof(CurrentThreadWin32) <= sizeof(ThreadControlData));
    CurrentThreadWin32& threadControlEx = (CurrentThreadWin32&) threadControl.data;

    constructDefault(threadControlEx);
    threadControl.intrface = &threadControlEx;

    return true;
}

//----------------------------------------------------------------

#endif
