#pragma once

#include "kit/kit.h"

//================================================================
//
// OverlayTakeoverID
//
//================================================================

class OverlayTakeoverID
{

public:

    OverlayTakeoverID(const OverlayTakeoverID&) =default;
    OverlayTakeoverID& operator =(const OverlayTakeoverID&) =default;

public:

    friend inline bool operator ==(const OverlayTakeoverID& a, const OverlayTakeoverID& b)
        {return a.value == b.value;}

public:

    static inline auto undefined() {return OverlayTakeoverID{size_t(-1)};}
    static inline auto cancelled() {return OverlayTakeoverID{size_t(-2)};}

public:

    explicit inline OverlayTakeoverID(size_t value)
        : value{value} {}

    size_t get() const {return value;}

private:

    size_t value;

};

//================================================================
//
// OverlayTakeover
//
// Only one user ID can be active.
//
//================================================================

struct OverlayTakeover
{
    virtual void setActiveID(const OverlayTakeoverID& id) const =0;
    virtual OverlayTakeoverID getActiveID() const =0;
};

//================================================================
//
// OverlayTakeoverKit
//
//================================================================

KIT_CREATE(OverlayTakeoverKit, const OverlayTakeover&, overlayTakeover);
