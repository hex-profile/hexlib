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

void threadCreate(ThreadFunc* threadFunc, void* threadParams, CpuAddrU stackSize, ThreadControl& threadControl, stdPars(ThreadToolKit));

//================================================================
//
// threadGetCurrent
//
//================================================================

void threadGetCurrent(ThreadControl& threadControl, stdPars(ThreadToolKit));

//================================================================
//
// mutexCreate
//
//================================================================

void mutexCreate(Mutex& section, stdPars(ThreadToolKit));

//================================================================
//
// eventCreate
//
//================================================================

void eventCreate(EventOwner& event, stdPars(ThreadToolKit));
