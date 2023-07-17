#pragma once

#include "cfg/cfgInterfaceFwd.h"
#include "storage/adapters/lambdaThunk.h"

//================================================================
//
// CfgSerialization
//
// Serialization visitor interface.
//
//================================================================

struct CfgSerialization
{
    virtual void operator()(const CfgSerializeKit& kit) =0;
};

////

LAMBDA_THUNK
(
    cfgSerializationThunk,
    CfgSerialization,
    void operator()(const CfgSerializeKit& kit),
    lambda(kit)
)
