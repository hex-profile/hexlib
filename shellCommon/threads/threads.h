#pragma once

#include "interfaces/syncObjects.h"
#include "errorLog/errorLogKit.h"
#include "stdFunc/stdFunc.h"
#include "storage/addrSpace.h"

//================================================================
//
// ThreadToolKit
//
//================================================================

using ThreadToolKit = ErrorLogKit;

//================================================================
//
// ThreadFunc
//
//================================================================

typedef void ThreadFunc(void* params);

//================================================================
//
// threadCreate
//
//================================================================

stdbool threadCreate(ThreadFunc* threadFunc, void* threadParams, CpuAddrU stackSize, ThreadControl& threadControl, stdPars(ThreadToolKit));

//================================================================
//
// threadGetCurrent
//
//================================================================

stdbool threadGetCurrent(ThreadControl& threadControl, stdPars(ThreadToolKit));

//================================================================
//
// mutexCreate
//
//================================================================

stdbool mutexCreate(Mutex& section, stdPars(ThreadToolKit));

//================================================================
//
// eventCreate
//
//================================================================

stdbool eventCreate(EventOwner& event, stdPars(ThreadToolKit));
