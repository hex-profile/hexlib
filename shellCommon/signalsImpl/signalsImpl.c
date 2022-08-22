#include "signalsImpl.h"

#include <string>

namespace signalImpl {

using namespace std;

//================================================================
//
// String
//
//================================================================

using String = basic_string<CharType>;

//================================================================
//
// GetStdString
//
// Can throw exceptions.
//
//================================================================

struct GetStdString : public CfgOutputString
{

public:

    inline GetStdString(String& result)
        : result(result) {}

    bool addBuf(const CharType* bufArray, size_t bufSize)
    {
        result.append(bufArray, bufSize);
        return true;
    }

private:

    String& result;

};

//================================================================
//
// getSignalNamePath
//
// Throws exceptions.
//
//================================================================

bool getSignalNamePath(const CfgNamespace* scope, const CfgSerializeSignal& signal, String& result)
{
    // Get main part
    GetStdString getName(result);
    ensure(signal.getName(getName));

    // Get namespace scope
    for (const CfgNamespace* p = scope; p != 0; p = p->prev)
        result = String(p->desc.ptr, p->desc.size) + CT("/") + result;

    return true;
}

//================================================================
//
// RegisterSignal
//
//================================================================

class RegisterSignal : public CfgVisitor
{

public:

    inline RegisterSignal(BaseActionSetup& registration)
        : registration(registration) {}

    void operator()(const CfgNamespace* scope, const CfgSerializeVariable& var)
        {}

    void operator()(const CfgNamespace* scope, const CfgSerializeSignal& signal)
    {
        try
        {
            String name;
            ensurev(getSignalNamePath(scope, signal, name));

            String key;
            GetStdString getKey(key);
            if_not (signal.getKey(getKey))
                key.clear();

            String comment;
            GetStdString getComment(comment);
            if_not (signal.getTextComment(getComment))
                comment.clear();

            uint32 id = currentId++;
            signal.setID(id);

            ensurev(registration.actsetAdd(id, key.c_str(), name.c_str(), comment.c_str()));
        }
        catch (const exception&) {}
    }

    int32 count() {return currentId - signalIdBase;}

private:

    BaseActionSetup& registration;
    uint32 currentId = signalIdBase;

};

//================================================================
//
// registerSignals
//
//================================================================

void registerSignals(CfgSerialization& serialization, const CfgNamespace* scope, BaseActionSetup& registration, int32& signalCount)
{
    RegisterSignal r(registration);
    serialization.serialize(CfgSerializeKit{r, scope});
    signalCount = r.count();
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

        uint32 signalIndex = id - signalIdBase;

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

        ////

        bool specialAction = (id >= baseActionId::LastPredefinedAction);


    };

    ////

    auto receiver = baseActionReceiverByLambda(handleAction);

    at.getActions(receiver);
}

//----------------------------------------------------------------

}
