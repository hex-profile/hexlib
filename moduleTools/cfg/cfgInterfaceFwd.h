#pragma once

#include "kit/kit.h"

//================================================================
//
// CfgSerializeKit
//
//================================================================

class CfgNamespace;
struct CfgVisitor;

//----------------------------------------------------------------

struct CfgSerializeKit
{
    CfgVisitor& visitor; 
    const CfgNamespace* scope;
};
