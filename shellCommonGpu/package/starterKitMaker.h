#pragma once

#include "allocation/mallocFlatAllocator/mallocFlatAllocator.h"
#include "compileTools/blockExceptionsSilent.h"
#include "debugBridge/bridgeUsage/msgLogToBridge.h"
#include "fileToolsImpl/fileToolsImpl.h"
#include "formattedOutput/diagLogTool.h"
#include "formattedOutput/messageFormatterStdio.h"
#include "package/starterKit.h"
#include "threading/threadManagerImpl.h"
#include "timerImpl/timerImpl.h"

namespace packageImpl {

//================================================================
//
// FormatterBuffer
//
//================================================================

class FormatterBuffer
{

public:

    operator Array<CharType> () const {return formatterArray;}

private:

    static constexpr size_t formatterSize = 65536;
    CharType formatterPtr[formatterSize];
    Array<CharType> formatterArray{formatterPtr, formatterSize};

};

//================================================================
//
// StarterKitMaker
//
// Field order matters.
//
//================================================================

struct StarterKitMaker
{
    StarterKitMaker(const Array<CharType>& formatterArray, DiagLog* diagLogPtr)
        : 
        formatter{formatterArray},
        diagLogPtr{diagLogPtr}
    {
    }

    ////

    MessageFormatterStdio formatter;

    DiagLog* const diagLogPtr;
    DiagLogNull diagLogNull;
    DiagLog& diagLog = diagLogPtr ? *diagLogPtr : diagLogNull;

    ////

    MsgLogByDiagLog msgLog{diagLog, formatter};
    MsgLogByDiagLog localLog{diagLog, formatter};

    ////

    ErrorLogByMsgLog errorLog{msgLog};
    ErrorLogExByMsgLog errorLogEx{msgLog};

    ////

    TimerImpl timer;
    FileToolsImpl fileTools;

    ////

    ErrorLogKit errorLogKit{errorLog};
    MAKE_MALLOC_ALLOCATOR_OBJECT(errorLogKit);

    ThreadManagerImpl threadManager;

    ////

    StarterKit kit = kitCombine
    (
        MessageFormatterKit(formatter),
        ErrorLogKit(errorLog),
        ErrorLogExKit(errorLogEx),
        MsgLogKit(msgLog),
        LocalLogKit(localLog),
        LocalLogAuxKit(false, localLog),
        TimerKit(timer),
        FileToolsKit(fileTools),
        MallocKit(mallocAllocator), 
        ThreadManagerKit(threadManager)
    );
};

//----------------------------------------------------------------

#define starterBegin \
    { \
        stdTraceRoot; \
        \
        StarterKitMaker kitMaker{formatterBuffer, kit.log}; \
        auto& kit = kitMaker.kit;

#define starterEnd \
        return true; \
    }

#define starterEndWith(value) \
        return value; \
    }

//================================================================
//
// StarterDebugKitMaker
//
// The field order matters.
//
//================================================================

struct StarterDebugKitMaker
{
    StarterDebugKitMaker(const Array<CharType>& formatterArray, bool flushEveryMessage, DiagLog* diagLogPtr, DebugBridge* debugBridgePtr, const DumpParams& dumpParams)
        : 
        formatter{formatterArray},
        flushEveryMessage{flushEveryMessage},
        diagLogPtr{diagLogPtr},
        debugBridgePtr{debugBridgePtr},
        dumpParams{dumpParams}
    {
        auto clear = [&] ()
        {
            debugBridge.localConsole()->clear();
        };

        blockExceptionsSilentVoid(clear());
    }

    ////

    ~StarterDebugKitMaker()
    {
        auto update = [&] ()
        {
            debugBridge.globalConsole()->update();
            debugBridge.localConsole()->update();
            debugBridge.videoOverlay()->update();
        };

        blockExceptionsSilentVoid(update());
    }

    ////

    MessageFormatterStdio formatter;
    bool const flushEveryMessage;
    
    DiagLog* const diagLogPtr;
    DiagLogNull diagLogNull;
    DiagLog& diagLog = diagLogPtr ? *diagLogPtr : diagLogNull;

    ////

    DebugBridge* const debugBridgePtr;
    DebugBridgeNull debugBridgeNull;
    DebugBridge& debugBridge = debugBridgePtr ? *debugBridgePtr : debugBridgeNull;

    DumpParams const dumpParams;

    ////

    MsgLogToDiagLogAndBridge msgLog{formatter, flushEveryMessage, diagLog, debugBridge.active(), *debugBridge.globalConsole()};
    MsgLogToDiagLogAndBridge localLog{formatter, flushEveryMessage, diagLog, debugBridge.active(), *debugBridge.localConsole()};

    ////

    ErrorLogByMsgLog errorLog{msgLog};
    ErrorLogExByMsgLog errorLogEx{msgLog};

    ////

    TimerImpl timer;
    FileToolsImpl fileTools;

    ////

    ErrorLogKit errorLogKit{errorLog};
    MAKE_MALLOC_ALLOCATOR_OBJECT(errorLogKit);

    ThreadManagerImpl threadManager;

    ////

    StarterDebugKit kit = kitCombine
    (
        MessageFormatterKit{formatter},
        ErrorLogKit{errorLog},
        ErrorLogExKit{errorLogEx},
        MsgLogKit{msgLog},
        LocalLogKit{localLog},
        LocalLogAuxKit{false, localLog},
        TimerKit{timer},
        FileToolsKit{fileTools},
        MallocKit{mallocAllocator}, 
        ThreadManagerKit{threadManager},
        DebugBridgeKit{debugBridge},
        DumpParamsKit{dumpParams}
    );
};

//----------------------------------------------------------------

#define starterDebugBegin(flushEveryMessage) \
    { \
        stdTraceRoot; \
        \
        StarterDebugKitMaker kitMaker{formatterBuffer, flushEveryMessage, kit.log, kit.debugBridge, kit.dumpParams}; \
        auto& kit = kitMaker.kit;

#define starterDebugEnd \
        return true; \
    }

#define starterDebugEndWith(value) \
        return value; \
    }

//----------------------------------------------------------------

}
