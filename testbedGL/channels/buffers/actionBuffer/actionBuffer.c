#include "actionBuffer.h"

#include "compileTools/blockExceptionsSilent.h"
#include "podVector/podVector.h"
#include "storage/rememberCleanup.h"

namespace actionBuffer {

//================================================================
//
// ActionBufferImpl
//
//================================================================

struct ActionBufferImpl : public ActionBuffer
{

    virtual void clearMemory()
    {
        cleared = false;
        buffer.dealloc();
    }

    //----------------------------------------------------------------
    //
    // Buffer API.
    //
    //----------------------------------------------------------------

    virtual bool hasUpdates() const
    {
        return
            cleared ||
            buffer.size() != 0;
    }

    virtual void reset()
    {
        cleared = false;
        buffer.clear();
    }

    virtual bool absorb(ActionBuffer& other);

    virtual void moveFrom(ActionBuffer& other);

    //----------------------------------------------------------------
    //
    // Data API.
    //
    //----------------------------------------------------------------

    virtual void dataClear()
    {
        cleared = true;
        buffer.clear();
    }

    ////

    virtual bool dataAdd(ActionId id)
    {
        boolFuncExceptBegin;

        buffer.push_back(id);

        boolFuncExceptEnd;
    }

    ////

    virtual size_t dataCount() const
    {
        return buffer.size();
    }

    ////

    virtual void dataGet(ActionIdReceiver& receiver) const
    {
        for (auto& r : buffer)
            receiver(r);
    }

    //----------------------------------------------------------------
    //
    // State.
    //
    //----------------------------------------------------------------

    bool cleared = false;

    PodVector<ActionId> buffer;

};

UniquePtr<ActionBuffer> ActionBuffer::create() {return makeUnique<ActionBufferImpl>();}

//================================================================
//
// ActionBufferImpl::absorb
//
//================================================================

bool ActionBufferImpl::absorb(ActionBuffer& other)
{
    boolFuncExceptBegin;

    auto& that = (ActionBufferImpl&) other;

    REMEMBER_CLEANUP(that.reset());

    //
    // Absorb clearing.
    //

    if (that.cleared)
        dataClear();

    //
    // Absorb buffer.
    //

    if (buffer.size() == 0)
    {
        buffer.swap(that.buffer);
    }
    else
    {
        buffer.append(that.buffer.data(), that.buffer.size(), false);
    }

    ////

    boolFuncExceptEnd;
}

//================================================================
//
// ActionBufferImpl::moveFrom
//
//================================================================

void ActionBufferImpl::moveFrom(ActionBuffer& other)
{
    auto& that = (ActionBufferImpl&) other;

    REMEMBER_CLEANUP(that.reset());

    ////

    this->cleared = that.cleared;

    this->buffer.swap(that.buffer); // no-throw
}

//----------------------------------------------------------------

}
