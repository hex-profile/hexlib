#pragma once

#include "cfgTools/overlayTakeover.h"

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

template <typename Target>
class OverlaySerializationThunk : public CfgSerialization
{

public:

    virtual void serialize(const CfgSerializeKit& kit)
        {target.serialize(kitCombine(kit, OverlayTakeoverKit{overlayTakeover}));}

    OverlaySerializationThunk(Target& target, OverlayTakeoverID& overlayOwnerID)
        : target(target), overlayTakeover(overlayOwnerID) {}

private:

    Target& target;
    OverlayTakeoverThunk overlayTakeover;

};

//----------------------------------------------------------------

template <typename Target>
inline auto overlaySerializationThunk(Target& target, OverlayTakeoverID& overlayOwnerID)
{
    return OverlaySerializationThunk<Target>(target, overlayOwnerID);
}
