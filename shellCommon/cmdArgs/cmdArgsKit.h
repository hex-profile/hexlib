#pragma once

#include "kit/kit.h"
#include "charType/charType.h"

//================================================================
//
// CmdArgsKit
//
//================================================================

KIT_CREATE2(CmdArgsKit, int, argCount, const CharType* const*, argStr);
