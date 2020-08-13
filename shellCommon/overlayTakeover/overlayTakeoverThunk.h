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

    void setActiveID(const ID& id) const
        {targetValue = id;}

    ID getActiveID() const
        {return targetValue;}

public:

    inline OverlayTakeoverThunk(ID& targetValue)
        : targetValue(targetValue) {}

private:

    ID& targetValue;

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

    OverlaySerializationThunk(Target& target, OverlayTakeover::ID& overlayOwnerID)
        : target(target), overlayTakeover(overlayOwnerID) {}

private:

    Target& target;
    OverlayTakeoverThunk overlayTakeover;

};

//----------------------------------------------------------------

template <typename Target>
inline auto overlaySerializationThunk(Target& target, OverlayTakeover::ID& overlayOwnerID)
{
    return OverlaySerializationThunk<Target>(target, overlayOwnerID);
}
