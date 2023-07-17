#pragma once

#include "kits/moduleKit.h"
#include "storage/adapters/callable.h"

//================================================================
//
// ModuleSerialization
//
// Extended serialization visitor interface.
//
//================================================================

using ModuleSerialization = Callable<void (const ModuleSerializeKit& kit)>;
