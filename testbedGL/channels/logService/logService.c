#include "logService.h"

#include <mutex>
#include <condition_variable>

namespace logService {

using namespace std;

//================================================================
//
// BoardImpl
//
//================================================================

struct BoardImpl : public Board
{
    //----------------------------------------------------------------
    //
    // Worker API.
    //
    //----------------------------------------------------------------

    virtual void takeAllUpdates(bool waitUpdates, const OptionalObject<uint32>& waitTimeoutMs, BufferRefs ownBuffers);

    //----------------------------------------------------------------
    //
    // GUI API.
    //
    //----------------------------------------------------------------

    virtual bool addAllUpdates(BufferRefs ownBuffers, bool notify);

    ////

    template <typename Buffer>
    bool addSpecificBuffer(Buffer& ownBuffer, Buffer& sharedBuffer);

    ////

    virtual bool addShutdownRequest(ShutdownBuffer& ownBuffer)
        {return addSpecificBuffer(ownBuffer, sharedBuffers.shutdownRequest);}

    virtual bool addTextUpdate(TextBuffer& ownBuffer)
        {return addSpecificBuffer(ownBuffer, sharedBuffers.textUpdate);}

    virtual bool addEditRequest(EditRequest& ownBuffer)
        {return addSpecificBuffer(ownBuffer, sharedBuffers.editRequest);}

    //----------------------------------------------------------------
    //
    // State.
    //
    //----------------------------------------------------------------

    mutex lock;
    condition_variable conditionVar;

    BufferInstances sharedBuffersMemory;
    BufferRefs sharedBuffers = sharedBuffersMemory.refs;

};

////

UniquePtr<Board> Board::create()
{
    return makeUnique<BoardImpl>();
}

//================================================================
//
// BoardImpl::takeAllUpdates
//
//================================================================

void BoardImpl::takeAllUpdates(bool waitUpdates, const OptionalObject<uint32>& waitTimeoutMs, BufferRefs ownBuffers)
{
    unique_lock<mutex> guard{lock};

    ////

    auto predicate = [&] {return sharedBuffers.hasUpdates();};

    if (waitUpdates)
    {
        if_not (waitTimeoutMs)
            conditionVar.wait(guard, predicate);
        else
        {
            using namespace std::chrono;
            conditionVar.wait_for(guard, milliseconds(*waitTimeoutMs), predicate);
        }
    }

    ownBuffers.moveFrom(sharedBuffers);
}

//================================================================
//
// BoardImpl::addAllUpdates
//
//================================================================

bool BoardImpl::addAllUpdates(BufferRefs ownBuffers, bool notify)
{
    if_not (ownBuffers.hasUpdates())
        return true;

    ////

    bool perfect = true;

    {
        unique_lock<mutex> guard{lock};

        perfect = sharedBuffers.absorb(ownBuffers);
    }

    ////

    if (notify)
        conditionVar.notify_one();

    ////

    return perfect;
}

//================================================================
//
// BoardImpl::addSpecificBuffer
//
//================================================================

template <typename Buffer>
bool BoardImpl::addSpecificBuffer(Buffer& ownBuffer, Buffer& sharedBuffer)
{
    if_not (ownBuffer.hasUpdates())
        return true;

    ////

    bool perfect = true;

    {
        unique_lock<mutex> guard{lock};

        perfect = sharedBuffer.absorb(ownBuffer);
    }

    ////

    conditionVar.notify_one();

    ////

    return perfect;
}

//----------------------------------------------------------------

}
