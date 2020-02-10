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
// OverlayTakeoverNull
//
//================================================================

class OverlayTakeoverNull : public OverlayTakeover
{
    void setActiveID(const ID& id) const
        {}

    ID getActiveID() const
        {return 0;}
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

    OverlaySerializationThunk(Target& target, const OverlayTakeover& overlayTakeover)
        : target(target), overlayTakeover(overlayTakeover) {}

private:

    Target& target;
    const OverlayTakeover& overlayTakeover;

};
