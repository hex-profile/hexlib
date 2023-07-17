#pragma once

#include "channels/buffers/small/editRequest.h"
#include "channels/buffers/small/shutdownBuffer.h"
#include "channels/buffers/textBuffer/textBuffer.h"
#include "compileTools/compileTools.h"
#include "numbers/int/intBase.h"
#include "storage/optionalObject.h"
#include "storage/smartPtr.h"

namespace logService {

//================================================================
//
// BufferRefs
//
//================================================================

struct BufferRefs
{
    ShutdownBuffer& shutdownRequest;
    TextBuffer& textUpdate;
    EditRequest& editRequest;

    ////

    template <typename Action>
    static sysinline void forEach(const BufferRefs& a, const BufferRefs& b, const Action& action)
    {
        action(a.shutdownRequest, b.shutdownRequest);
        action(a.textUpdate, b.textUpdate);
        action(a.editRequest, b.editRequest);
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
    UniqueInstance<TextBuffer> textUpdate;
    UniqueInstance<EditRequest> editRequest;

    BufferRefs refs = {*shutdownRequest, *textUpdate, *editRequest};
};

//================================================================
//
// ServerApi
//
//================================================================

struct ServerApi
{
    virtual void takeAllUpdates(bool waitUpdates, const OptionalObject<uint32>& waitTimeoutMs, BufferRefs ownBuffers) =0;
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
    virtual bool addTextUpdate(TextBuffer& ownBuffer) =0;
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
