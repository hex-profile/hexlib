#include "signalsImpl.h"

#include <string>

#include "at_client.h"

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

    inline RegisterSignal(AtSignalSet& registration)
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

            ensurev(registration.actsetAdd(id, name.c_str(), key.c_str(), comment.c_str()));
        }
        catch (const exception&) {}
    }

    int32 count() {return currentId - signalIdBase;}

private:

    AtSignalSet& registration;
    uint32 currentId = signalIdBase;

};

//================================================================
//
// registerSignals
//
//================================================================

void registerSignals(CfgSerialization& serialization, const CfgNamespace* scope, AtSignalSet& registration, int32& signalCount)
{
    RegisterSignal r(registration);
    serialization.serialize(CfgSerializeKit(r, scope));
    signalCount = r.count();
}

//================================================================
//
// prepareSignalHistogram
//
//================================================================

void prepareSignalHistogram(AtSignalTest& at, const Array<int32>& histogram, bool& anyEventsFound, bool& realEventsFound, bool& mouseSignal, bool& mouseSignalAlt)
{
    ARRAY_EXPOSE(histogram);

    anyEventsFound = false;
    realEventsFound = false;
    mouseSignal = false;
    mouseSignalAlt = false;

    //
    // Zero array
    //

    for_count (i, histogramSize)
        histogramPtr[i] = 0;

    //
    // Collect action counts
    //

    int32 actionCount = at.actionCount();

    for_count (i, actionCount)
    {
        AtActionId id = 0;

        if (at.actionItem(i, id))
        {
            uint32 k = id - signalIdBase;

            if (SpaceU(k) < SpaceU(histogramSize))
                ++histogramPtr[k];
        }

        if (id == AT_ACTION_ID_MOUSE_LEFT_DOWN)
            mouseSignal = true;

        if (id == AT_ACTION_ID_MOUSE_RIGHT_DOWN)
            mouseSignalAlt = true;

        ////

        anyEventsFound = true;

        if_not
        (
            id == AT_ACTION_ID_MOUSE_LEFT_UP ||
            id == AT_ACTION_ID_MOUSE_RIGHT_UP
        )
        {
            realEventsFound = true;
        }
    }
}

//----------------------------------------------------------------

}
