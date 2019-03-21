#pragma once

#include "storage/opaqueStruct.h"
#include "prepTools/prepBase.h"
#include "dataAlloc/deallocInterface.h"
#include "storage/destructibleInterface.h"
#include "storage/constructDestruct.h"

//================================================================
//
// CriticalSection
//
//================================================================

using CriticalSectionData = OpaqueStruct<64>;

//----------------------------------------------------------------

struct CriticalSectionInterface
{
    virtual void enter() =0;
    virtual void leave() =0;
    virtual bool tryEnter() =0;

    virtual ~CriticalSectionInterface() {}
};

//----------------------------------------------------------------

struct CriticalSection
{

public:

    inline CriticalSectionInterface* operator->()
        {return intrface;}

public:

    inline bool created()
        {return intrface != 0;}

    inline void clear()
        {if (intrface) intrface->~CriticalSectionInterface(); intrface = 0;}

    inline CriticalSection()
        : intrface(0) {}

    inline ~CriticalSection()
        {clear();}

public:

    CriticalSectionInterface* intrface;
    CriticalSectionData data;

};

//================================================================
//
// CritsecGuard
//
//================================================================

class CritsecGuard
{

    CriticalSection& section;

public:

    inline CritsecGuard(CriticalSection& section)
        :
        section(section)
    {
        section->enter();
    }

    inline ~CritsecGuard()
    {
        section->leave();
    }
};

//----------------------------------------------------------------

#define CRITSEC_GUARD(section) \
    CritsecGuard PREP_PASTE2(__guard_, __LINE__)(section)

//================================================================
//
// Event
//
// Wait operation resets the event state.
//
//================================================================

using EventData = OpaqueStruct<16>;

//----------------------------------------------------------------

struct Event
{
    virtual void set() =0;
    virtual void reset() =0;

    virtual void wait() =0;
    virtual bool waitWithTimeout(uint32 timeMs) =0;
};

//----------------------------------------------------------------

struct EventOwner
{

public:

    inline operator Event& ()
        {return *intrface;}

    inline void set()
        {intrface->set();}

    inline void reset()
        {intrface->reset();}

    inline void wait()
        {intrface->wait();}

    inline bool waitWithTimeout(uint32 timeMs)
        {return intrface->waitWithTimeout(timeMs);}

public:

    inline bool created()
        {return intrface != 0;}

    inline void clear()
        {if (intrface) destruct(*intrface); intrface = 0;}

    inline EventOwner()
        : intrface(0) {}

    inline ~EventOwner()
        {clear();}

public:

    DestructibleInterface<Event>* intrface;
    EventData data;

};

//================================================================
//
// ThreadPriority
//
//================================================================

enum ThreadPriority
{
    ThreadPriorityMinusMax,
    ThreadPriorityMinus2,
    ThreadPriorityMinus1,
    ThreadPriorityNormal,
    ThreadPriorityPlus1,
    ThreadPriorityPlus2,
    ThreadPriorityPlusMax
};

//================================================================
//
// ThreadControl
//
//================================================================

using ThreadControlData = OpaqueStruct<64>;

//----------------------------------------------------------------

struct ThreadControlInterface
{
    virtual bool setPriority(ThreadPriority priority) =0;

    virtual ~ThreadControlInterface() {}
};

//----------------------------------------------------------------

struct ThreadControl
{

public:

    inline ThreadControlInterface* operator->()
        {return intrface;}

public:

    inline bool created()
        {return intrface != 0;}

    inline void waitAndClear()
        {if (intrface) intrface->~ThreadControlInterface(); intrface = 0;}

    inline ThreadControl()
        : intrface(0) {}

    inline ~ThreadControl()
        {waitAndClear();}

public:

    ThreadControlInterface* intrface;
    ThreadControlData data;

};
