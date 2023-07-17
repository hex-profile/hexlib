#pragma once

#include "charType/charArray.h"
#include "storage/smartPtr.h"
#include "storage/adapters/callable.h"
#include "baseInterfaces/actionDefs.h"

namespace actionBuffer {

//================================================================
//
// ActionIdReceiver
//
//================================================================

using ActionIdReceiver = Callable<void (ActionId id)>;

//================================================================
//
// ActionBuffer
//
//================================================================

struct ActionBuffer
{
    static UniquePtr<ActionBuffer> create();
    virtual ~ActionBuffer() {}

    virtual void clearMemory() =0;

    //
    // Data API.
    //

    virtual void dataClear() =0;

    virtual bool dataAdd(ActionId id) =0;

    virtual size_t dataCount() const =0;

    virtual void dataGet(ActionIdReceiver& receiver) const =0;

    //
    // Buffer API.
    //

    virtual bool hasUpdates() const =0;

    virtual void reset() =0;

    virtual bool absorb(ActionBuffer& other) =0;

    virtual void moveFrom(ActionBuffer& other) =0;
};

//----------------------------------------------------------------

}

using actionBuffer::ActionBuffer;
using actionBuffer::ActionIdReceiver;
