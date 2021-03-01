#pragma once

#include "stdFunc/stdFunc.h"

namespace cfgVarsImpl {

//================================================================
//
// StringReceiver
//
//================================================================

struct StringReceiver
{
    virtual stdbool receive(const CharArray& str, stdNullPars) =0;
};

//----------------------------------------------------------------

}
