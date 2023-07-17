#include "guiService.h"

#include <mutex>

#include "compileTools/compileTools.h"
#include "errorLog/debugBreak.h"

namespace guiService {

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

    virtual bool addAllUpdates(BufferRefs ownBuffers);

    ////

    template <typename Buffer>
    bool addSpecificBuffer(Buffer& buffer, Buffer& sharedBuffer);

    ////

    virtual bool addShutdownRequest(ShutdownBuffer& ownBuffer)
        {return addSpecificBuffer(ownBuffer, sharedBuffers.shutdownRequest);}

    virtual bool addGlobalLogUpdate(LogBuffer& ownBuffer)
        {return addSpecificBuffer(ownBuffer, sharedBuffers.globalLogUpdate);}

   virtual bool addLocalLogUpdate(LogBuffer& ownBuffer)
        {return addSpecificBuffer(ownBuffer, sharedBuffers.localLogUpdate);}

   virtual bool addSpecialLogUpdate(LogBuffer& ownBuffer)
        {return addSpecificBuffer(ownBuffer, sharedBuffers.specialLogUpdate);}

    virtual bool addActionSetUpdate(ActionSetBuffer& ownBuffer)
        {return addSpecificBuffer(ownBuffer, sharedBuffers.actionSetUpdate);}

    virtual bool addOverlayUpdate(OverlayBuffer& ownBuffer)
        {return addSpecificBuffer(ownBuffer, sharedBuffers.overlayUpdate);}

    virtual bool addConfigUpdate(CfgTree& ownBuffer)
        {return addSpecificBuffer(ownBuffer, sharedBuffers.configUpdate);}

    //----------------------------------------------------------------
    //
    // GUI API.
    //
    //----------------------------------------------------------------

    virtual bool checkUpdates();

    virtual void takeAllUpdates(BufferRefs dst);

    ////

    virtual void takeOverlayUpdate(OverlayBuffer& ownBuffer)
    {
        unique_lock<mutex> guard{lock};

        ownBuffer.moveFrom(sharedBuffers.overlayUpdate);
    }

    //----------------------------------------------------------------
    //
    // The signaller callback.
    //
    //----------------------------------------------------------------

    Signaller* theSignaller = nullptr;

    virtual void setSignaller(Signaller* signaller)
        {theSignaller = signaller;}

    //----------------------------------------------------------------
    //
    // Shared state.
    //
    //----------------------------------------------------------------

    mutex lock;

    BufferInstances sharedBuffersMemory;
    BufferRefs sharedBuffers = sharedBuffersMemory.refs;

};

////

UniquePtr<Board> Board::create()
    {return makeUnique<BoardImpl>();}

//================================================================
//
// BoardImpl::addAllUpdates
//
//================================================================

bool BoardImpl::addAllUpdates(BufferRefs ownBuffers)
{
    if_not (ownBuffers.hasUpdates())
        return true;

    ////

    bool signalled{};
    bool perfect = true;

    {
        unique_lock<mutex> guard{lock};

        signalled = sharedBuffers.hasUpdates();

        perfect = sharedBuffers.absorb(ownBuffers);
    }

    ////

    if_not (signalled)
    {
        if (theSignaller)
            (*theSignaller)();
    }

    ////

    return perfect;
}

//================================================================
//
// BoardImpl::addSpecificBuffer
//
//================================================================

template <typename Buffer>
bool BoardImpl::addSpecificBuffer(Buffer& buffer, Buffer& sharedBuffer)
{
    if_not (buffer.hasUpdates())
        return true;

    ////

    bool perfect = true;
    bool signalled{};

    {
        unique_lock<mutex> guard{lock};

        signalled = sharedBuffers.hasUpdates();

        perfect = sharedBuffer.absorb(buffer);
    }

    ////

    if_not (signalled)
    {
        if (theSignaller)
            (*theSignaller)();
    }

    ////

    return perfect;
}

//================================================================
//
// BoardImpl::checkUpdates
//
//================================================================

bool BoardImpl::checkUpdates()
{
    unique_lock<mutex> guard{lock};

    return sharedBuffers.hasUpdates();
}

//================================================================
//
// BoardImpl::takeAllUpdates
//
//================================================================

void BoardImpl::takeAllUpdates(BufferRefs ownBuffers)
{
    {
        unique_lock<mutex> guard{lock};

        ////

        ownBuffers.moveFrom(sharedBuffers);

        DEBUG_BREAK_CHECK(!sharedBuffers.hasUpdates());
    }
}

//----------------------------------------------------------------

}
