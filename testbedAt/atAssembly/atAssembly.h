#pragma once

#include "allocation/mallocKit.h"
#include "atInterface/atInterfaceKit.h"
#include "interfaces/threadManagerKit.h"
#include "kits/moduleHeader.h"
#include "storage/dynamicClass.h"
#include "atEngine/atEngine.h"

namespace atStartup {

//================================================================
//
// AtAssembly
//
// The main stateful module for AT shell.
//
//================================================================

//================================================================
//
// InitKit
// ProcessKit
//
//================================================================

KIT_COMBINE6(InitKit, ErrorLogKit, ErrorLogExKit, MsgLogsKit, AtCommonKit, MallocKit, ThreadManagerKit);
KIT_COMBINE6(ProcessKit, ErrorLogKit, ErrorLogExKit, MsgLogsKit, AtProcessKit, MallocKit, ThreadManagerKit);

//================================================================
//
// AtAssembly
//
//================================================================

class AtAssembly
{

public:

    AtAssembly();
    ~AtAssembly();

public:

    stdbool init(const AtEngineFactory& engineFactory, stdPars(InitKit));
    stdvoid finalize(stdPars(InitKit));

    stdbool process(stdPars(ProcessKit));

private:

    DynamicClass<class AtAssemblyImpl> instance;

};

//----------------------------------------------------------------

}
