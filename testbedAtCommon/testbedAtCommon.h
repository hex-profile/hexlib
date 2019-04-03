#pragma once

#include "at_client.h"
#include "atEngine/atEngine.h"

//================================================================
//
// atClientCreate
// atClientDestroy
// atClientProcess
//
//================================================================

void atClientCreate(void** instance, const at_api_create* api, const AtEngineFactory& engineFactory);
void atClientDestroy(void* instance, const at_api_destroy* api);
void atClientProcess(void* instance, const at_api_process* api);
