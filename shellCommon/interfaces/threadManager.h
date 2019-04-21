#pragma once

#include "interfaces/syncObjects.h"
#include "errorLog/errorLogKit.h"
#include "stdFunc/stdFunc.h"
#include "threadManagerKit.h"
#include "storage/addrSpace.h"

//================================================================
//
// ThreadToolKit
//
//================================================================

KIT_COMBINE1(ThreadToolKit, ErrorLogKit);

//================================================================
//
// ThreadFunc
//
//================================================================

typedef void ThreadFunc(void* params);

//================================================================
//
// ThreadManager
//
//================================================================

struct ThreadManager
{

    virtual stdbool createCriticalSection(CriticalSection& section, stdPars(ThreadToolKit)) =0;

    virtual stdbool createEvent(bool manualReset, EventOwner& event, stdPars(ThreadToolKit)) =0;

    virtual stdbool createThread(ThreadFunc* threadFunc, void* threadParams, CpuAddrU stackSize, ThreadControl& threadControl, stdPars(ThreadToolKit)) =0;

    virtual stdbool getCurrentThread(ThreadControl& threadControl, stdPars(ThreadToolKit)) =0;

};
