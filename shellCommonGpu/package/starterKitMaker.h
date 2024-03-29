#pragma once

#include "allocation/mallocAllocator/mallocAllocator.h"
#include "compileTools/blockExceptionsSilent.h"
#include "debugBridge/bridgeUsage/msgLogToBridge.h"
#include "formattedOutput/diagLogTool.h"
#include "formattedOutput/formatters/messageFormatterImpl.h"
#include "package/starterKit.h"
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
    Array<CharType> formatterArray = makeArray(formatterPtr, formatterSize);

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

    MessageFormatterImpl formatter;

    DiagLog* const diagLogPtr;
    DiagLogNull diagLogNull;
    DiagLog& diagLog = diagLogPtr ? *diagLogPtr : diagLogNull;

    ////

    MsgLogByDiagLog msgLog{diagLog, formatter};
    MsgLogByDiagLog localLog{diagLog, formatter};

    ////

    ErrorLogByMsgLog errorLog{msgLog};
    MsgLogExByMsgLog msgLogEx{msgLog};

    ////

    TimerImpl timer;

    ////

    ErrorLogKit errorLogKit{errorLog};
    MAKE_MALLOC_ALLOCATOR(errorLogKit);

    ////

    StarterKit kit = kitCombine
    (
        MessageFormatterKit(formatter),
        ErrorLogKit(errorLog),
        MsgLogExKit(msgLogEx),
        MsgLogKit(msgLog),
        LocalLogKit(localLog),
        LocalLogAuxKit(false, localLog),
        TimerKit(timer),
        MallocKit(mallocAllocator)
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
        blockExceptBegin;
        debugBridge.statusConsole()->clear();
        blockExceptEndIgnore;
    }

    ////

    ~StarterDebugKitMaker()
    {
        blockExceptBegin;
        debugBridge.commit();
        blockExceptEndIgnore;
    }

    ////

    MessageFormatterImpl formatter;
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

    MsgLogToDiagLogAndBridge msgLog{formatter, flushEveryMessage, diagLog, debugBridge.active(), *debugBridge.messageConsole()};
    MsgLogToDiagLogAndBridge localLog{formatter, flushEveryMessage, diagLog, debugBridge.active(), *debugBridge.statusConsole()};

    ////

    ErrorLogByMsgLog errorLog{msgLog};
    MsgLogExByMsgLog msgLogEx{msgLog};

    ////

    TimerImpl timer;

    ////

    ErrorLogKit errorLogKit{errorLog};
    MAKE_MALLOC_ALLOCATOR(errorLogKit);

    ////

    StarterDebugKit kit = kitCombine
    (
        MessageFormatterKit{formatter},
        ErrorLogKit{errorLog},
        MsgLogExKit{msgLogEx},
        MsgLogKit{msgLog},
        LocalLogKit{localLog},
        LocalLogAuxKit{false, localLog},
        TimerKit{timer},
        MallocKit{mallocAllocator},
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
