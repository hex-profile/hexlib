#pragma once

#include "channels/buffers/small/editRequest.h"
#include "channels/buffers/small/shutdownBuffer.h"
#include "cfgVars/cfgTree/cfgTree.h"
#include "storage/smartPtr.h"

namespace configService {

//================================================================
//
// BufferRefs
//
//================================================================

struct BufferRefs
{
    ShutdownBuffer& shutdownRequest;
    CfgTree& configUpdate;
    EditRequest& editRequest;

    void clearMemory() const
    {
        shutdownRequest.clearMemory();
        configUpdate.clearMemory();
        editRequest.clearMemory();
    }

    bool hasUpdates() const
    {
        return
            shutdownRequest.hasUpdates() ||
            configUpdate.hasUpdates() ||
            editRequest.hasUpdates();
    }

    void reset() const
    {
        shutdownRequest.reset();
        configUpdate.reset();
        editRequest.reset();
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
        ok &= configUpdate.absorb(other.configUpdate);
        ok &= editRequest.absorb(other.editRequest);

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
        configUpdate.moveFrom(other.configUpdate);
        editRequest.moveFrom(other.editRequest);
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
    UniqueInstance<CfgTree> configUpdate;
    UniqueInstance<EditRequest> editRequest;

    BufferRefs refs = {*shutdownRequest, *configUpdate, *editRequest};
};

//================================================================
//
// ServerApi
//
//================================================================

struct BoardDiagnostics
{
    size_t configBufferBytes = 0;
};

////

struct ServerApi
{
    virtual void takeAllUpdates(bool waitUpdates, const OptionalObject<uint32>& waitTimeoutMs, BufferRefs ownBuffers, BoardDiagnostics& diagnostics) =0;
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
    virtual bool addConfigUpdate(CfgTree& ownBuffer) =0;
    virtual bool addEditRequest(EditRequest& ownBuffer) =0;

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
