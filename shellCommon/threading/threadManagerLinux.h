
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

    bool createCriticalSection(CriticalSection& section, stdPars(ThreadToolKit));

    bool createEvent(bool manualReset, EventOwner& event, stdPars(ThreadToolKit));

    bool createThread(ThreadFunc* threadFunc, void* threadParams, CpuAddrU stackSize, ThreadControl& threadControl, stdPars(ThreadToolKit));

    bool getCurrentThread(ThreadControl& threadControl, stdPars(ThreadToolKit));

};
