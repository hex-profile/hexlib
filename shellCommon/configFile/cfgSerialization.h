#pragma once

#include "cfg/cfgInterfaceFwd.h"

//================================================================
//
// CfgSerialization
//
// Serialization visitor interface
//
//================================================================

struct CfgSerialization
{
    virtual void serialize(const CfgSerializeKit& kit) =0;
};
