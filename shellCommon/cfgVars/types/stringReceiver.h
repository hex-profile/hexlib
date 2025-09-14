#pragma once

#include "stdFunc/stdFunc.h"
#include "storage/adapters/callable.h"

namespace cfgVarsImpl {

//================================================================
//
// StringReceiver
//
//================================================================

using StringReceiver = Callable<void (const CharArray& str, stdParsNull)>;

//----------------------------------------------------------------

}
