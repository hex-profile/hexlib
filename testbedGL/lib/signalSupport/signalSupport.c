#include "signalSupport.h"

#include "cfg/cfgInterface.h"
#include "compileTools/classContext.h"
#include "errorLog/debugBreak.h"
#include "stlString/stlString.h"
#include "userOutput/printMsgEx.h"
#include "lib/signalSupport/parseKey.h"
#include "dataAlloc/arrayObjectMemory.inl"
#include "lib/signalSupport/printSignalKey.h"

//================================================================
//
// ParseKey
//
//================================================================

struct ParseKey : public CfgOutputString
{
    bool addBuf(const CharType* bufArray, size_t bufSize)
    {
        ++partCount;
        return parseKey(CharArray(bufArray, bufSize), keyResult);
    }

    KeyRec keyResult;
    Space partCount = 0;
};

//================================================================
//
// SignalSupport::~SignalSupport
//
//================================================================

SignalSupport::~SignalSupport()
{
}

//================================================================
//
// SignalSupport::initSignals
//
//================================================================

stdbool SignalSupport::initSignals(CfgSerialization& serialization, stdPars(InitKit))
{

    //----------------------------------------------------------------
    //
    // Count signals
    //
    //----------------------------------------------------------------

    Space signalCount = 0;

    auto enumSignals = cfgVisitSignal | [&] (auto& signal)
    {
        signal.setID(signalCount);
        ++signalCount;
    };

    serialization({CfgVisitVarNull{}, enumSignals, CfgScopeVisitorNull{}});

    //----------------------------------------------------------------
    //
    // Allocate signal space
    //
    //----------------------------------------------------------------

    require(signalKeys.reallocInHeap(signalCount, stdPass));

    //----------------------------------------------------------------
    //
    // Parse keys
    //
    //----------------------------------------------------------------

    auto parseSignal = cfgVisitSignal | [&] (auto& signal)
    {
        ParseKey parseKey;
        bool ok = signal.getKey(parseKey);

        check_flag(parseKey.partCount == 1, ok);

        if_not (ok)
        {
            printMsg(kit.msgLog, STR("Cannot parse key name \"%0\""), PrintSignalKey{signal}, msgWarn);
            return;
        }

        ////

        auto& keys = signalKeys;
        ARRAY_EXPOSE(keys);

        Space id = signal.getID();
        ensurev(DEBUG_BREAK_CHECK(SpaceU(id) < SpaceU(keysSize)));

        keysPtr[id] = parseKey.keyResult;
    };

    serialization({CfgVisitVarNull{}, parseSignal, CfgScopeVisitorNull{}});

    ////

    returnTrue;
}

//================================================================
//
// SignalSupport::feedSignals
//
//================================================================

void SignalSupport::feedSignals(const Array<const KeyEvent>& keys, CfgSerialization& serialization)
{
    if_not (hasData(keys))
    {
        auto resetSignal = cfgVisitSignal | [&] (auto& signal)
        {
            signal.setImpulseCount(0);
        };

        serialization({CfgVisitVarNull{}, resetSignal, CfgScopeVisitorNull{}});
    }
    else
    {
        auto feedSignal = cfgVisitSignal | [&] (auto& signal)
        {
            ARRAY_EXPOSE(signalKeys);
            Space id = signal.getID();
            ensurev(DEBUG_BREAK_CHECK(SpaceU(id) < SpaceU(signalKeysSize)));

            ////

            auto& signalKey = signalKeysPtr[id];

            ARRAY_EXPOSE(keys);

            int32 impulseCount = 0;

            for_count (i, keysSize)
            {
                auto& key = keysPtr[i];

                if
                (
                    key.action != KeyAction::Release &&
                    key.code != 0 &&
                    key.code == signalKey.code &&
                    key.modifiers == signalKey.modifiers
                )
                    ++impulseCount;
            }

            signal.setImpulseCount(impulseCount);
        };

        serialization({CfgVisitVarNull{}, feedSignal, CfgScopeVisitorNull{}});
    }
}
