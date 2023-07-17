#pragma once

#include "storage/smartPtr.h"
#include "channels/buffers/actionBuffer/actionBuffer.h"
#include "channels/buffers/small/displaySettingsBuffer.h"
#include "channels/buffers/small/mousePointerBuffer.h"
#include "channels/buffers/small/shutdownBuffer.h"
#include "cfgVars/cfgTree/cfgTree.h"

namespace workerService {

//================================================================
//
// BufferRefs
//
//================================================================

struct BufferRefs
{
    ShutdownBuffer& shutdownRequest;
    ActionBuffer& actionsUpdate;
    DisplaySettingsBuffer& displaySettingsUpdate;
    MousePointerBuffer& mousePointerUpdate;
    CfgTree& configUpdate;

    void clearMemory() const
    {
        shutdownRequest.clearMemory();
        actionsUpdate.clearMemory();
        displaySettingsUpdate.clearMemory();
        mousePointerUpdate.clearMemory();
        configUpdate.clearMemory();
    }

    bool hasUpdates() const
    {
        return
            shutdownRequest.hasUpdates() ||
            actionsUpdate.hasUpdates() ||
            displaySettingsUpdate.hasUpdates() ||
            mousePointerUpdate.hasUpdates() ||
            configUpdate.hasUpdates();
    }

    void reset() const
    {
        shutdownRequest.reset();
        actionsUpdate.reset();
        displaySettingsUpdate.reset();
        mousePointerUpdate.reset();
        configUpdate.reset();
    }

    bool absorb(BufferRefs other) const
    {
        if_not (other.hasUpdates())
            return true;

        if_not (hasUpdates())
        {
            moveFrom(other);
            return true;
        }

        bool ok = true;

        ok &= shutdownRequest.absorb(other.shutdownRequest);
        ok &= actionsUpdate.absorb(other.actionsUpdate);
        ok &= displaySettingsUpdate.absorb(other.displaySettingsUpdate);
        ok &= mousePointerUpdate.absorb(other.mousePointerUpdate);
        ok &= configUpdate.absorb(other.configUpdate);

        return ok;
    }

    void moveFrom(BufferRefs other) const
    {
        if_not (other.hasUpdates())
        {
            reset();
            return;
        }

        shutdownRequest.moveFrom(other.shutdownRequest);
        actionsUpdate.moveFrom(other.actionsUpdate);
        displaySettingsUpdate.moveFrom(other.displaySettingsUpdate);
        mousePointerUpdate.moveFrom(other.mousePointerUpdate);
        configUpdate.moveFrom(other.configUpdate);
    }
};

//================================================================
//
// BufferInstances
//
// Convenience class.
//
//================================================================

struct BufferInstances
{
    UniqueInstance<ShutdownBuffer> shutdownRequest;
    UniqueInstance<ActionBuffer> actionsUpdate;
    UniqueInstance<DisplaySettingsBuffer> displaySettingsUpdate;
    UniqueInstance<MousePointerBuffer> mousePointerUpdate;
    UniqueInstance<CfgTree> configUpdate;

    BufferRefs refs = {*shutdownRequest, *actionsUpdate, *displaySettingsUpdate, *mousePointerUpdate, *configUpdate};
};

//================================================================
//
// ServerApi
//
//================================================================

struct ServerApi
{
    virtual void takeAllUpdates(bool waitUpdates, BufferRefs ownBuffers) =0;
};

//================================================================
//
// ClientApi
//
// The returned bool flag means "update is completely successful",
// otherwise update could be partially successful.
//
//================================================================

struct ClientApi
{
    virtual bool addShutdownRequest(ShutdownBuffer& ownBuffer) =0;
    virtual bool addActionsUpdate(ActionBuffer& ownBuffer) =0;
    virtual bool addDisplaySettingsUpdate(DisplaySettingsBuffer& ownBuffer) =0;
    virtual bool addMousePointerUpdate(MousePointerBuffer& ownBuffer) =0;
    virtual bool addConfigUpdate(CfgTree& ownBuffer) =0;

    virtual bool addAllUpdates(BufferRefs ownBuffers, bool notify) =0;
};

//================================================================
//
// Board
//
//================================================================

struct Board : public ServerApi, public ClientApi
{
    static UniquePtr<Board> create();
    virtual ~Board() {}
};

//----------------------------------------------------------------

}
