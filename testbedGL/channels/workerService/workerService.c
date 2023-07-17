#include "workerService.h"

#include <mutex>
#include <condition_variable>

#include "compileTools/compileTools.h"

namespace workerService {

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

    virtual void takeAllUpdates(bool waitUpdates, BufferRefs dst);

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

    virtual bool addActionsUpdate(ActionBuffer& ownBuffer)
        {return addSpecificBuffer(ownBuffer, sharedBuffers.actionsUpdate);}

    virtual bool addDisplaySettingsUpdate(DisplaySettingsBuffer& ownBuffer)
        {return addSpecificBuffer(ownBuffer, sharedBuffers.displaySettingsUpdate);}

    virtual bool addMousePointerUpdate(MousePointerBuffer& ownBuffer)
        {return addSpecificBuffer(ownBuffer, sharedBuffers.mousePointerUpdate);}

    virtual bool addConfigUpdate(CfgTree& ownBuffer)
        {return addSpecificBuffer(ownBuffer, sharedBuffers.configUpdate);}

    //----------------------------------------------------------------
    //
    // State.
    //
    //----------------------------------------------------------------

    mutex lock;
    condition_variable condition;

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

void BoardImpl::takeAllUpdates(bool waitUpdates, BufferRefs ownBuffers)
{
    unique_lock<mutex> guard{lock};

    if (waitUpdates)
        condition.wait(guard, [&] {return sharedBuffers.hasUpdates();});

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
        condition.notify_one();

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

    condition.notify_one();

    ////

    return perfect;
}

//----------------------------------------------------------------

}
