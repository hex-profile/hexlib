#pragma once

#include "stdFunc/stdFunc.h"
#include "errorLog/errorLogKit.h"
#include "userOutput/msgLogExKit.h"

//================================================================
//
// ContextBinder
//
// Set/unset TLS variables.
//
//================================================================

struct ContextBinder
{
    virtual ~ContextBinder() {}

    using Kit = KitCombine<ErrorLogKit, MsgLogExKit>;

    virtual void bind(stdPars(Kit)) =0;
    virtual void unbind(stdPars(Kit)) =0;
};
