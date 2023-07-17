#pragma once

#include "cfgTools/overlayTakeover.h"
#include "cfg/cfgSerialization.h"

//================================================================
//
// OverlayTakeoverThunk
//
//================================================================

class OverlayTakeoverThunk : public OverlayTakeover
{

public:

    void setActiveID(const OverlayTakeoverID& id) const
        {targetValue = id;}

    OverlayTakeoverID getActiveID() const
        {return targetValue;}

public:

    inline OverlayTakeoverThunk(OverlayTakeoverID& targetValue)
        : targetValue(targetValue) {}

private:

    OverlayTakeoverID& targetValue;

};

//================================================================
//
// OverlaySerializationThunk
//
//================================================================

template <typename Lambda>
class OverlaySerializationThunk : public CfgSerialization
{

public:

    virtual void operator()(const CfgSerializeKit& kit)
        {lambda(kitCombine(kit, OverlayTakeoverKit{overlayTakeover}));}

    OverlaySerializationThunk(const Lambda& lambda, OverlayTakeoverID& overlayOwnerID)
        : lambda(lambda), overlayTakeover(overlayOwnerID) {}

private:

    Lambda lambda;
    OverlayTakeoverThunk overlayTakeover;

};

//----------------------------------------------------------------

template <typename Lambda>
inline auto overlaySerializationThunk(const Lambda& lambda, OverlayTakeoverID& overlayOwnerID)
{
    return OverlaySerializationThunk<Lambda>(lambda, overlayOwnerID);
}
