
#pragma once

#include "interfaces/threadManager.h"

//================================================================
//
// ThreadManagerLinux
//
//================================================================

class ThreadManagerLinux : public ThreadManager
{

public:

    stdbool createCriticalSection(CriticalSection& section, stdPars(ThreadToolKit));

    stdbool createEvent(bool manualReset, EventOwner& event, stdPars(ThreadToolKit));

    stdbool createThread(ThreadFunc* threadFunc, void* threadParams, CpuAddrU stackSize, ThreadControl& threadControl, stdPars(ThreadToolKit));

    stdbool getCurrentThread(ThreadControl& threadControl, stdPars(ThreadToolKit));

};
