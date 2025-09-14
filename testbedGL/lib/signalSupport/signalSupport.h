#pragma once

#include "allocation/mallocKit.h"
#include "compileTools/classContext.h"
#include "cfg/cfgSerialization.h"
#include "dataAlloc/arrayObjectMemory.h"
#include "lib/keys/keyBase.h"
#include "stdFunc/stdFunc.h"
#include "userOutput/diagnosticKit.h"

//================================================================
//
// SignalSupport
//
//================================================================

class SignalSupport
{

public:

    ~SignalSupport();

public:

    using InitKit = KitCombine<MallocKit, DiagnosticKit>;
    void initSignals(CfgSerialization& serialization, stdPars(InitKit));

public:

    void feedSignals(const Array<const KeyEvent>& keys, CfgSerialization& serialization);

private:

    ArrayObjectMemory<KeyRec> signalKeys;

};
