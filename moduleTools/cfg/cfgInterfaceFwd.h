#pragma once

#include "storage/opaqueStructFwd.h"

//================================================================
//
// CfgScopeContext
//
// Specific data of a scope visitor.
//
//================================================================

using CfgScopeContext = OpaqueStruct<4 * sizeof(void*), 0x3CA8C539u>;

//================================================================
//
// CfgSerializeKit
//
//================================================================

struct CfgVisitVar;
struct CfgVisitSignal;
struct CfgScopeVisitor;

//----------------------------------------------------------------

struct CfgSerializeKit
{
    const CfgVisitVar& visitVar;
    const CfgVisitSignal& visitSignal;
    const CfgScopeVisitor& scopeVisitor;
};
