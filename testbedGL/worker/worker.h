#pragma once

#include "storage/smartPtr.h"
#include "cfg/cfgSerialization.h"
#include "channels/workerService/workerService.h"
#include "channels/guiService/guiService.h"
#include "stdFunc/stdFunc.h"
#include "testModule/testModule.h"
#include "minimalShell/minimalShellTypes.h"
#include "userOutput/diagnosticKit.h"
#include "allocation/mallocKit.h"
#include "channels/configService/configService.h"
#include "lib/contextBinder.h"

namespace worker {

using minimalShell::GpuExternalContext;

//================================================================
//
// ContextBinderGL
//
// Set/unset OpenGL TLS variables.
//
//================================================================

struct ContextBinderGL
{
    virtual void bind(stdParsNull) =0;
    virtual void unbind(stdParsNull) =0;
};

//================================================================
//
// ExternalsFactory
//
//================================================================

using ExternalsFactory = Callable<UniquePtr<TestModuleExternals> ()>;

//================================================================
//
// Worker
//
//================================================================

struct Worker
{
    static UniquePtr<Worker> create(UniquePtr<TestModule>&& testModule);
    virtual ~Worker() {}

    //----------------------------------------------------------------
    //
    // Config.
    //
    //----------------------------------------------------------------

    virtual void serialize(const CfgSerializeKit& kit) =0;

    //----------------------------------------------------------------
    //
    // Init.
    //
    //----------------------------------------------------------------

    struct InitArgs
    {
        ContextBinder& glContext; // The class keeps a pointer to it!
        ExternalsFactory& externalsFactory;
        guiService::ClientApi& guiService;
    };

    using InitKit = KitCombine<DiagnosticKit, MallocKit>;

    virtual void init(const InitArgs& args, stdPars(InitKit)) =0;

    //----------------------------------------------------------------
    //
    // Run.
    //
    //----------------------------------------------------------------

    struct RunArgs
    {
        GpuExternalContext externalContext;

        workerService::ServerApi& workerService;
        guiService::ClientApi& guiService;
        configService::ClientApi& configService;
    };

    virtual void run(const RunArgs& args) =0;

};

//----------------------------------------------------------------

}

using worker::Worker;
