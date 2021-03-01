#pragma once

#include "baseInterfaces/baseSignals.h"
#include "cfg/cfgInterface.h"
#include "cfg/cfgInterfaceFwd.h"
#include "configFile/cfgSerialization.h"
#include "data/array.h"
#include "cfgTools/overlayTakeover.h"
#include "cfgTools/standardSignal.h"

namespace signalImpl {

//================================================================
//
// registerSignals
//
//================================================================

void registerSignals(CfgSerialization& serialization, const CfgNamespace* scope, BaseActionSetup& registration, int32& signalCount);

//================================================================
//
// SignalsOverview
//
//================================================================

struct SignalsOverview
{
    bool anyEventsFound;
    bool realEventsFound;
    bool mouseSignal;
    bool mouseSignalAlt;
    bool saveConfig;
    bool loadConfig;
    bool editConfig;
};

//================================================================
//
// prepareSignalHistogram
//
//================================================================

void prepareSignalHistogram(BaseActionReceiving& at, const Array<int32>& histogram, SignalsOverview& overview);

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

//================================================================
//
// handleSignals
//
//================================================================

template <typename Assembly>
inline void handleSignals(Assembly& that, const Array<const int32>& signalHist, OverlayTakeover::ID& overlayOwnerID, StandardSignal& deactivateOverlay)
{
    auto prevOverlayID = overlayOwnerID;

    {
        FeedSignal visitor(signalHist);
        that.serialize(CfgSerializeKit(visitor, nullptr));
    }

    //
    // Deactivate overlay
    //

    if (deactivateOverlay)
        overlayOwnerID = 0;

    //
    // If overlay owner has changed, re-feed all the signals
    // to clean outdated switches.
    //

    if (prevOverlayID != overlayOwnerID)
    {
        FeedSignal visitor(signalHist);
        that.serialize(CfgSerializeKit(visitor, nullptr));
    }
}

//----------------------------------------------------------------

}
