#include "legacySignalsImpl.h"

#include <stdexcept>

#include "podVector/stringStorage.h"
#include "storage/rememberCleanup.h"
#include "errorLog/convertExceptions.h"

namespace signalImpl {

using namespace std;

//================================================================
//
// StringStorage
//
//================================================================

using StringStorage = StringStorageEx<CharType>;

//================================================================
//
// registerSignals
//
//================================================================

void registerSignals(CfgSerialization& serialization, BaseActionSetup& registration, int32& signalCount, stdPars(Kit))
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

    signalCount = 0;

    auto visitSignal = cfgVisitSignal | [&] (auto& signal)
    {
        auto id = signalCount++;
        signal.setID(id);

        ////

        auto savedSize = path.size();
        REMEMBER_CLEANUP(path.resize(savedSize, false));

        ////

        if (path.size())
            path.append(STR("/"));

        signal.getName(appendToPath);

        path.push_back(0);

        ////

        key.clear();
        signal.getKey(appendToKey);
        key.push_back(0);

        ////

        comment.clear();
        signal.getTextComment(appendToComment);
        comment.push_back(0);

        ////

        if_not (registration.actsetAdd(id, key.data(), path.data(), comment.data()))
            throw std::bad_alloc();
    };

    ////

    serialization({CfgVisitVarNull{}, visitSignal, scopeVisitor});

    stdExceptEnd;
}

//================================================================
//
// BaseActionReceiverByLambda
//
//================================================================

template <typename Lambda>
class BaseActionReceiverByLambda : public BaseActionReceiver
{

public:

    BaseActionReceiverByLambda(const Lambda& lambda)
        : lambda{lambda} {}

    virtual void process(const Array<const BaseActionRec>& actions)
    {
        ARRAY_EXPOSE(actions);

        for_count (i, actionsSize)
            lambda(actionsPtr[i]);
    }

private:

    Lambda lambda;

};

//----------------------------------------------------------------

template <typename Lambda>
inline auto baseActionReceiverByLambda(const Lambda& lambda)
    {return BaseActionReceiverByLambda<Lambda>{lambda};}

//================================================================
//
// prepareSignalHistogram
//
//================================================================

void prepareSignalHistogram(BaseActionReceiving& at, const Array<int32>& histogram, SignalsOverview& overview)
{
    ARRAY_EXPOSE(histogram);

    ////

    overview = SignalsOverview{};

    //
    // Zero array
    //

    for_count (i, histogramSize)
        histogramPtr[i] = 0;

    //
    // Collect action counts
    //

    auto handleAction = [&] (const BaseActionRec& rec) -> void
    {
        auto id = rec.id;

        ////

        if (rec.mousePos.valid())
            overview.mousePos = rec.mousePos;

        ////

        uint32 signalIndex = id;

        if (SpaceU(signalIndex) < SpaceU(histogramSize))
            ++histogramPtr[signalIndex];

        ////

        if (id == baseActionId::MouseLeftDown)
            overview.mouseLeftSet++;

        if (id == baseActionId::MouseRightDown)
            overview.mouseRightSet++;

        if (id == baseActionId::MouseLeftUp)
            overview.mouseLeftReset++;

        if (id == baseActionId::MouseRightUp)
            overview.mouseRightReset++;

        ////

        if (id == baseActionId::SaveConfig)
            overview.saveConfig = true;

        if (id == baseActionId::LoadConfig)
            overview.loadConfig = true;

        if (id == baseActionId::EditConfig)
            overview.editConfig = true;

        ////

        if (id == baseActionId::ResetupActions)
            overview.resetupActions = true;

        ////

        overview.anyEventsFound = true;
        overview.realEventsFound = true;
    };

    ////

    auto receiver = baseActionReceiverByLambda(handleAction);

    at.getActions(receiver);
}

//----------------------------------------------------------------

}
