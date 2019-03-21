#pragma once

#include "numbers/int/intBase.h"
#include "kit/kit.h"

//================================================================
//
// OverlayTakeover
//
// Only one user ID can be active.
//
//================================================================

struct OverlayTakeover
{
    virtual void setActiveID(uint32 id) =0;
    virtual uint32 getActiveID() =0;
};

//================================================================
//
// OverlayTakeoverNull
//
//================================================================

class OverlayTakeoverNull : public OverlayTakeover
{
    void setActiveID(uint32 id)
        {}

    uint32 getActiveID()
        {return 0;}
};

//================================================================
//
// OverlayTakeoverKit
//
//================================================================

KIT_CREATE1(OverlayTakeoverKit, OverlayTakeover&, overlayTakeover);
