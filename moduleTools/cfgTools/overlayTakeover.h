#pragma once

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
    virtual void setActiveID(size_t id) =0;
    virtual size_t getActiveID() =0;
};

//================================================================
//
// OverlayTakeoverNull
//
//================================================================

class OverlayTakeoverNull : public OverlayTakeover
{
    void setActiveID(size_t id)
        {}

    size_t getActiveID()
        {return 0;}
};

//================================================================
//
// OverlayTakeoverKit
//
//================================================================

KIT_CREATE1(OverlayTakeoverKit, OverlayTakeover&, overlayTakeover);
