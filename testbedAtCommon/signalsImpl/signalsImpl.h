#pragma once

#include "cfg/cfgInterfaceFwd.h"
#include "data/array.h"
#include "atInterface/atInterface.h"
#include "configFile/cfgSerialization.h"
#include "cfg/cfgInterface.h"

namespace signalImpl {

//================================================================
//
// registerSignals
//
//================================================================

void registerSignals(CfgSerialization& serialization, const CfgNamespace* scope, AtSignalSet& registration, int32& signalCount);

//================================================================
//
// prepareSignalHistogram
//
//================================================================

void prepareSignalHistogram(AtSignalTest& at, const Array<int32>& histogram, bool& anyEventsFound, bool& realEventsFound, bool& mouseSignal, bool& mouseSignalAlt);

//================================================================
//
// signalIdBase
//
//================================================================

const uint32 signalIdBase = 0x3111BA5E;

//================================================================
//
// FeedSignal
//
//================================================================

class FeedSignal : public CfgVisitor
{

public:

    inline FeedSignal(const Array<const int32>& histogram)
        : histogram(histogram) {}

    void operator()(const CfgNamespace* scope, const CfgSerializeVariable& var)
        {}

    void operator()(const CfgNamespace* scope, const CfgSerializeSignal& signal)
    {
        ARRAY_EXPOSE(histogram);

        uint32 id = signal.getID() - signalIdBase;

        int32 impulseCount = 0;

        if (SpaceU(id) < SpaceU(histogramSize))
            impulseCount = histogramPtr[id];

        signal.setImpulseCount(impulseCount);
    }

private:

    Array<const int32> histogram;

};

//----------------------------------------------------------------

}
