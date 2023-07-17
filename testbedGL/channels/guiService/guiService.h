#pragma once

#include "channels/buffers/small/shutdownBuffer.h"
#include "channels/buffers/logBuffer/logBuffer.h"
#include "channels/buffers/actionSetBuffer/actionSetBuffer.h"
#include "channels/buffers/overlayBuffer/overlayBuffer.h"
#include "cfgVars/cfgTree/cfgTree.h"

namespace guiService {

//================================================================
//
// BufferRefs
//
//================================================================

struct BufferRefs
{
    ShutdownBuffer& shutdownRequest;
    LogBuffer& globalLogUpdate;
    LogBuffer& localLogUpdate;
    LogBuffer& specialLogUpdate; // Log updater messages.
    ActionSetBuffer& actionSetUpdate;
    OverlayBuffer& overlayUpdate;
    CfgTree& configUpdate;

    ////

    template <typename Action>
    static sysinline void forEach(const BufferRefs& a, const BufferRefs& b, const Action& action)
    {
        action(a.shutdownRequest, b.shutdownRequest);
        action(a.globalLogUpdate, b.globalLogUpdate);
        action(a.localLogUpdate, b.localLogUpdate);
        action(a.specialLogUpdate, b.specialLogUpdate);
        action(a.actionSetUpdate, b.actionSetUpdate);
        action(a.overlayUpdate, b.overlayUpdate);
        action(a.configUpdate, b.configUpdate);
    }

    ////

    void clearMemory() const
    {
        auto action = [&] (auto& p, auto&)
            {p.clearMemory();};

        forEach(*this, *this, action);
    }

    ////

    bool hasUpdates() const
    {
        bool result = false;

        auto action = [&] (auto& p, auto&)
            {result |= p.hasUpdates();};

        forEach(*this, *this, action);

        return result;
    }

    ////

    void reset() const
    {
        auto action = [&] (auto& p, auto&)
            {p.reset();};

        forEach(*this, *this, action);
    }

    ////

    bool absorb(const BufferRefs& other) const
    {
        if_not (other.hasUpdates())
            return true;

        if_not (hasUpdates())
        {
            moveFrom(other);
            return true;
        }

        bool ok = true;

        auto action = [&] (auto& a, auto& b)
            {ok &= a.absorb(b);};

        forEach(*this, other, action);

        return ok;
    }

    ////

    void moveFrom(BufferRefs other) const
    {
        if_not (other.hasUpdates())
        {
            reset();
            return;
        }

        auto action = [&] (auto& a, auto& b)
            {a.moveFrom(b);};

        forEach(*this, other, action);
    }
};

////

inline BufferRefs replaceGlobalLog(const BufferRefs& that, LogBuffer& globalLog)
{
    return
    {
        that.shutdownRequest,
        globalLog,
        that.localLogUpdate,
        that.specialLogUpdate,
        that.actionSetUpdate,
        that.overlayUpdate,
        that.configUpdate
    };
}

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
    UniqueInstance<LogBuffer> globalLogUpdate;
    UniqueInstance<LogBuffer> localLogUpdate;
    UniqueInstance<LogBuffer> specialLogUpdate;
    UniqueInstance<ActionSetBuffer> actionSetUpdate;
    UniqueInstance<OverlayBuffer> overlayUpdate;
    UniqueInstance<CfgTree> configUpdate;

    BufferRefs refs =
    {
        *shutdownRequest,
        *globalLogUpdate,
        *localLogUpdate,
        *specialLogUpdate,
        *actionSetUpdate,
        *overlayUpdate,
        *configUpdate
    };
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
    virtual bool addGlobalLogUpdate(LogBuffer& ownBuffer) =0;
    virtual bool addLocalLogUpdate(LogBuffer& ownBuffer) =0;
    virtual bool addSpecialLogUpdate(LogBuffer& ownBuffer) =0;
    virtual bool addActionSetUpdate(ActionSetBuffer& ownBuffer) =0;
    virtual bool addOverlayUpdate(OverlayBuffer& ownBuffer) =0;
    virtual bool addConfigUpdate(CfgTree& ownBuffer) =0;

    virtual bool addAllUpdates(BufferRefs ownBuffers) =0;
};

//================================================================
//
// ServerApi
//
//================================================================

struct ServerApi
{
    virtual bool checkUpdates() =0;

    virtual void takeAllUpdates(BufferRefs ownBuffers) =0;

    virtual void takeOverlayUpdate(OverlayBuffer& ownBuffer) =0;
};

//================================================================
//
// Signaller
//
//================================================================

using Signaller = Callable<void ()>; // Async thread-proof

//================================================================
//
// Board
//
//================================================================

struct Board : public ClientApi, public ServerApi
{
    //----------------------------------------------------------------
    //
    // Init / deinit.
    //
    // To be used before threads are started,
    // including the setting of a signaller setting: it is not protected.
    //
    //----------------------------------------------------------------

    static UniquePtr<Board> create();
    virtual ~Board() {}

    virtual void setSignaller(Signaller* signaller) =0;
};

//----------------------------------------------------------------

}
