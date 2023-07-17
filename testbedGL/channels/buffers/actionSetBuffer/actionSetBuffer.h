#pragma once

#include "charType/charArray.h"
#include "storage/smartPtr.h"
#include "storage/adapters/callable.h"
#include "baseInterfaces/actionDefs.h"

namespace actionSetBuffer {

//================================================================
//
// ActionRecordReceiver
//
//================================================================

using ActionRecordReceiver = Callable<void (ActionId id, CharArray name, CharArray key, CharArray comment)>;

//================================================================
//
// ActionSetBuffer
//
//================================================================

struct ActionSetBuffer
{
    static UniquePtr<ActionSetBuffer> create();
    virtual ~ActionSetBuffer() {}

    virtual void clearMemory() =0;

    //
    // Data API.
    //

    virtual void dataClear() =0;

    virtual bool dataAdd(ActionId id, CharArray name, CharArray key, CharArray comment) =0;

    virtual size_t dataCount() const =0;

    virtual void dataGet(ActionRecordReceiver& receiver) const =0;

    //
    // Buffer API.
    //

    virtual bool hasUpdates() const =0;

    virtual void reset() =0;

    virtual bool absorb(ActionSetBuffer& other) =0;

    virtual void moveFrom(ActionSetBuffer& other) =0;
};

//----------------------------------------------------------------

}

using actionSetBuffer::ActionSetBuffer;
using actionSetBuffer::ActionRecordReceiver;
