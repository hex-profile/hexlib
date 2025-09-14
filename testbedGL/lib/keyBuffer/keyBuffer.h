#pragma once

#include "lib/keys/keyBase.h"
#include "storage/smartPtr.h"
#include "stdFunc/stdFunc.h"
#include "data/array.h"
#include "errorLog/errorLogKit.h"
#include "userOutput/msgLogExKit.h"

//================================================================
//
// KeyBufferImpl
//
//================================================================

struct KeyBuffer
{
    static UniquePtr<KeyBuffer> create();
    virtual ~KeyBuffer() {}

    ////

    using Kit = KitCombine<ErrorLogKit, MsgLogExKit>;

    ////

    virtual void reserve(Space capacity, stdPars(Kit)) =0;
    virtual void dealloc() =0;

    ////

    virtual void receiveKey(const KeyEvent& key, stdPars(Kit)) =0;

    ////

    virtual Array<const KeyEvent> getBuffer() const =0;
    virtual void clearBuffer() =0;
};
