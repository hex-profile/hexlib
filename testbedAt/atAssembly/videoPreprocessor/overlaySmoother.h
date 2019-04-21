#pragma once

#include "kits/moduleHeader.h"
#include "atInterface/atInterface.h"
#include "allocation/mallocKit.h"
#include "interfaces/threadManagerKit.h"

namespace overlaySmoother {

//================================================================
//
// InitKit
//
//================================================================

KIT_COMBINE2(InitKit, ErrorLogKit, ThreadManagerKit);

//================================================================
//
// ProcessKit
//
//================================================================

KIT_COMBINE2(ProcessKit, ModuleProcessKit, MallocKit);

//================================================================
//
// OverlaySmoother
//
//================================================================

class OverlaySmoother
{

public:

    OverlaySmoother();
    ~OverlaySmoother();

    stdbool init(stdPars(InitKit));
    void deinit();

public:

    stdbool setImage(const Point<Space>& size, AtImageProvider<uint8_x4>& imageProvider, const FormatOutputAtom& desc, uint32 id, bool textEnabled, stdPars(ProcessKit));
    stdbool setFakeImage(stdPars(ProcessKit)) {return true;}
    stdbool updateImage(stdPars(ProcessKit));
    stdbool clearQueue(stdPars(ProcessKit));
    stdbool setSmoothing(bool smoothing, stdPars(ProcessKit));
    stdbool flushSmoothly(stdPars(ProcessKit));

public:

    void serialize(const ModuleSerializeKit& kit);
    void setOutputInterface(AtAsyncOverlay* output);

    AtVideoOverlay* getInputInterface();

private:

    StaticClass<class OverlaySmootherImpl, 1 << 13> instance;

};

//================================================================
//
// OverlaySmootherThunk
//
//================================================================

class OverlaySmootherThunk : public AtVideoOverlay
{

public:

    OverlaySmootherThunk(OverlaySmoother& base, const ProcessKit& kit)
        : base(base), kit(kit) {}

public:

    bool setImage(const Point<Space>& size, AtImageProvider<uint8_x4>& imageProvider, const FormatOutputAtom& desc, uint32 id, bool textEnabled, stdNullPars)
        {overlayIsSet = true; return base.setImage(size, imageProvider, desc, id, textEnabled, stdPassThru);}

    bool setFakeImage(stdNullPars)
        {overlayIsSet = true; return base.setFakeImage(stdPassThru);}

    bool updateImage(stdNullPars)
        {return base.updateImage(stdPassThru);}

    bool clearQueue(stdNullPars)
        {return base.clearQueue(stdPassThru);}

    bool setSmoothing(bool smoothing, stdNullPars)
        {return base.setSmoothing(smoothing, stdPassThru);}

    bool flushSmoothly(stdNullPars)
        {return base.flushSmoothly(stdPassThru);}

public:

    bool overlayIsSet = false;

private:

    OverlaySmoother& base;
    ProcessKit const kit;

};

//----------------------------------------------------------------

}
