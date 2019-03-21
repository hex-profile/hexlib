#pragma once

#include "cfg/cfgInterfaceFwd.h"
#include "configFile/cfgStringEnv.h"
#include "configFile/cfgSerialization.h"

namespace cfgVarsImpl {

//================================================================
//
// Cfgvar set implementation using STL for format I/O
// and loading/storing to StringEnv interface.
//
//================================================================

void saveVarsToStringEnv(CfgSerialization& serialization, const CfgNamespace* scope, StringEnv& stringEnv);
void loadVarsFromStringEnv(CfgSerialization& serialization, const CfgNamespace* scope, StringEnv& stringEnv);

//----------------------------------------------------------------

}
