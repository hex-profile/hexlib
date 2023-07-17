#pragma once

#include "storage/opaqueStruct.h"
#include "prepTools/prepBase.h"
#include "dataAlloc/deallocInterface.h"
#include "storage/constructDestruct.h"

//================================================================
//
// Mutex
//
// * Fast, works only inside a single process, like critical section.
// * Second lock from the same thread may NOT be supported!
//
//================================================================

using MutexData = OpaqueStruct<64, 0x66CE270Eu>;

//----------------------------------------------------------------

struct MutexInterface
{
    virtual void lock() =0;
    virtual void unlock() =0;
    virtual bool tryLock() =0;

    virtual ~MutexInterface() {}
};

//----------------------------------------------------------------

struct Mutex
{

public:

    inline void lock()
        {return intrface->lock();}

    inline void unlock()
        {return intrface->unlock();}

    inline bool tryLock()
        {return intrface->tryLock();}

public:

    inline bool created()
        {return intrface != 0;}

    inline void clear()
        {if (intrface) intrface->~MutexInterface(); intrface = 0;}

    inline Mutex()
        : intrface(0) {}

    inline ~Mutex()
        {clear();}

public:

    MutexInterface* intrface;
    MutexData data;

};

//================================================================
//
// MutexGuard
//
//================================================================

class MutexGuard
{

    Mutex& base;

public:

    inline MutexGuard(Mutex& base)
        :
        base(base)
    {
        base.lock();
    }

    inline ~MutexGuard()
    {
        base.unlock();
    }
};

//----------------------------------------------------------------

#define MUTEX_GUARD(base) \
    MutexGuard PREP_PASTE2(__guard_, __LINE__)(base)

//================================================================
//
// Event
//
// Wait operation resets the event state.
//
//================================================================

using EventData = OpaqueStruct<24, 0x595234CBu>;

//----------------------------------------------------------------

struct EventInterface
{
    virtual void set() =0;
    virtual void wait() =0;
    virtual bool waitWithTimeout(uint32 timeMs) =0;

    virtual ~EventInterface() {}
};

//----------------------------------------------------------------

struct EventOwner
{

public:

    inline void set()
        {return intrface->set();}

    inline void wait()
        {return intrface->wait();}

    bool waitWithTimeout(uint32 timeMs)
        {return intrface->waitWithTimeout(timeMs);}

public:

    inline bool created()
        {return intrface != 0;}

    inline void clear()
        {if (intrface) intrface->~EventInterface(); intrface = 0;}

    inline EventOwner()
        : intrface(0) {}

    inline ~EventOwner()
        {clear();}

public:

    EventInterface* intrface;
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

using ThreadControlData = OpaqueStruct<64, 0x513F8C8Bu>;

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
