#include "signalTools.h"

#include "cfg/cfgInterface.h"
#include "compileTools/blockExceptionsSilent.h"
#include "podVector/stringStorage.h"
#include "storage/rememberCleanup.h"
#include "errorLog/convertExceptions.h"
#include "errorLog/debugBreak.h"

namespace signalTools {

using namespace std;

//================================================================
//
// StringStorage
//
//================================================================

using StringStorage = StringStorageEx<CharType>;

//================================================================
//
// gatherActionSet
//
//================================================================

stdbool gatherActionSet(CfgSerialization& serialization, ActionReceiver& receiver, size_t& actionCount, stdPars(Kit))
{
    stdExceptBegin;

    //----------------------------------------------------------------
    //
    // String vars.
    //
    //----------------------------------------------------------------

    StringStorage path;
    path.reserve(500);

    StringStorage key;
    key.reserve(50);

    StringStorage comment;
    comment.reserve(200);

    ////

    auto appendToPath = cfgOutputString | [&] (auto ptr, auto size)
        {path.append(ptr, size, true); return true;};

    auto appendToKey = cfgOutputString | [&] (auto ptr, auto size)
        {key.append(ptr, size, true); return true;};

    auto appendToComment = cfgOutputString | [&] (auto ptr, auto size)
        {comment.append(ptr, size, true); return true;};

    //----------------------------------------------------------------
    //
    // Scope visitor.
    //
    //----------------------------------------------------------------

    struct ScopeContext
    {
        size_t savedSize = 0;
    };

    ////

    auto enter = [&] (auto& context, auto& name)
    {
        auto& the = context.template recast<ScopeContext>();
        the.savedSize = path.size();

        if (path.size())
            path.append(STR("/"));

        path.append(name);
    };


    ////

    auto leave = [&] (auto& context)
    {
        auto& the = context.template recast<ScopeContext>();

        path.resize(the.savedSize, false);
    };

    ////

    auto scopeVisitor = cfgScopeVisitor(enter, leave);

    //----------------------------------------------------------------
    //
    // Enumerate signals and pass to the receiver.
    //
    //----------------------------------------------------------------

    uint32 signalCount = 0;

    auto visitSignal = cfgVisitSignal | [&] (auto& signal)
    {
        auto id = signalCount++;
        signal.setID(id);

        ////

        auto savedSize = path.size();
        REMEMBER_CLEANUP(path.resize(savedSize, false));

        if (path.size())
            path.append(STR("/"));

        signal.getName(appendToPath);

        key.clear();
        signal.getKey(appendToKey);

        comment.clear();
        signal.getTextComment(appendToComment);

        receiver(id, path, key, comment);
    };

    ////

    serialization({CfgVisitVarNull{}, visitSignal, scopeVisitor});

    ////

    actionCount = signalCount;

    stdExceptEnd;
}

//================================================================
//
// updateSignals
//
//================================================================

void updateSignals(bool providerHasData, ActionIdProvider& provider, CfgSerialization& serialization, const Array<int32>& actionHist)
{
    if_not (providerHasData)
    {
        auto resetSignal = cfgVisitSignal | [&] (auto& signal)
        {
            signal.setImpulseCount(0);
        };

        serialization({CfgVisitVarNull{}, resetSignal, CfgScopeVisitorNull{}});
    }
    else
    {
        ARRAY_EXPOSE(actionHist);

        for_count (i, actionHistSize)
            actionHistPtr[i] = 0;

        ////

        auto receiver = ActionIdReceiver::O | [&] (ActionId id)
        {
            if (SpaceU(id) < SpaceU(actionHistSize))
                actionHistPtr[id] += 1;
        };

        provider(receiver);

        ////

        auto feedSignal = cfgVisitSignal | [&] (auto& signal)
        {
            auto id = signal.getID();

            int32 count = 0;

            if (SpaceU(id) < SpaceU(actionHistSize))
                count = actionHistPtr[id];

            signal.setImpulseCount(count);
        };

        serialization({CfgVisitVarNull{}, feedSignal, CfgScopeVisitorNull{}});
    }
}

//----------------------------------------------------------------

}
