#pragma once

#include "stdFunc/stdFunc.h"
#include "storage/adapters/callable.h"

namespace cfgVarsImpl {

//================================================================
//
// StringReceiver
//
//================================================================

using StringReceiver = Callable<stdbool (const CharArray& str, stdParsNull)>;

//----------------------------------------------------------------

}
