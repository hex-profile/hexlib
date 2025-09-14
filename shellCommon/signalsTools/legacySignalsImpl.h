#pragma once

#include "baseInterfaces/baseSignals.h"
#include "cfgTools/overlayTakeover.h"
#include "cfgTools/standardSignal.h"
#include "cfg/cfgSerialization.h"
#include "data/array.h"
#include "userOutput/msgLogExKit.h"
#include "stdFunc/stdFunc.h"

namespace signalImpl {

//================================================================
//
// Kit
//
//================================================================

using Kit = MsgLogExKit;

//================================================================
//
// registerSignals
//
//================================================================

void registerSignals(CfgSerialization& serialization, BaseActionSetup& registration, int32& signalCount, stdPars(Kit));

//================================================================
//
// SignalsOverview
//
//================================================================

struct SignalsOverview
{
    bool anyEventsFound = false;
    bool realEventsFound = false;

    BaseMousePos mousePos;
    int32 mouseLeftSet = 0;
    int32 mouseLeftReset = 0;
    int32 mouseRightSet = 0;
    int32 mouseRightReset = 0;

    bool saveConfig = false;
    bool loadConfig = false;
    bool editConfig = false;

    bool resetupActions = false;
};

//================================================================
//
// prepareSignalHistogram
//
//================================================================

void prepareSignalHistogram(BaseActionReceiving& at, const Array<int32>& histogram, SignalsOverview& overview);

//================================================================
//
// handleSignals
//
//================================================================

inline void handleSignals(CfgSerialization& serialization, const Array<const int32>& signalHist, OverlayTakeoverID& overlayOwnerID, StandardSignal& deactivateOverlay)
{
    auto prevOverlayID = overlayOwnerID;

    //
    // Feed signal.
    //

    auto histogram = signalHist;

    auto feedSignal = cfgVisitSignal | [&] (auto& signal)
    {
        ARRAY_EXPOSE(histogram);

        uint32 id = signal.getID();

        int32 impulseCount = 0;

        if (SpaceU(id) < SpaceU(histogramSize))
            impulseCount = histogramPtr[id];

        signal.setImpulseCount(impulseCount);
    };

    ////

    serialization({CfgVisitVarNull{}, feedSignal, CfgScopeVisitorNull{}});

    //
    // Deactivate overlay
    //

    if (deactivateOverlay)
        overlayOwnerID = OverlayTakeoverID::cancelled();

    //
    // If overlay owner has changed, re-feed all the signals
    // to clean outdated switches.
    //

    if_not (prevOverlayID == overlayOwnerID)
        serialization({CfgVisitVarNull{}, feedSignal, CfgScopeVisitorNull{}});
}

//----------------------------------------------------------------

}
