#pragma once

#include "kit/kit.h"

//================================================================
//
// CfgSerializeKit
//
//================================================================

class CfgNamespace;
struct CfgVisitor;

KIT_CREATE2(CfgSerializeKit, CfgVisitor&, visitor, const CfgNamespace*, scope);
