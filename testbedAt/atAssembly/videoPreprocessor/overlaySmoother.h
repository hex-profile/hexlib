#pragma once

#include "kits/moduleHeader.h"
#include "atInterface/atInterface.h"
#include "allocation/mallocKit.h"

namespace overlaySmoother {

//================================================================
//
// InitKit
//
//================================================================

using InitKit = ErrorLogKit;

//================================================================
//
// ProcessKit
//
//================================================================

using ProcessKit = KitCombine<ModuleProcessKit, MallocKit>;

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

    stdbool setImage(const Point<Space>& size, BaseImageProvider& imageProvider, const FormatOutputAtom& desc, uint32 id, bool textEnabled, stdPars(ProcessKit));
    stdbool setImageFake(stdPars(ProcessKit)) {returnTrue;}
    stdbool updateImage(stdPars(ProcessKit));
    stdbool clearQueue(stdPars(ProcessKit));
    stdbool setSmoothing(bool smoothing, stdPars(ProcessKit));
    stdbool flushSmoothly(stdPars(ProcessKit));

public:

    void serialize(const ModuleSerializeKit& kit);
    void setOutputInterface(AtAsyncOverlay* output);

    BaseVideoOverlay* getInputInterface();

private:

    DynamicClass<class OverlaySmootherImpl> instance;

};

//================================================================
//
// OverlaySmootherThunk
//
//================================================================

class OverlaySmootherThunk : public BaseVideoOverlay
{

public:

    OverlaySmootherThunk(OverlaySmoother& base, const ProcessKit& kit)
        : base(base), kit(kit) {}

public:

    stdbool setImage(const Point<Space>& size, bool dataProcessing, BaseImageProvider& imageProvider, const FormatOutputAtom& desc, uint32 id, bool textEnabled, stdParsNull)
        {overlayIsSet = true; return base.setImage(size, imageProvider, desc, id, textEnabled, stdPassThru);}

    stdbool setImageFake(stdParsNull)
        {overlayIsSet = true; return base.setImageFake(stdPassThru);}

    stdbool updateImage(stdParsNull)
        {return base.updateImage(stdPassThru);}

    stdbool clearQueue(stdParsNull)
        {return base.clearQueue(stdPassThru);}

    stdbool setSmoothing(bool smoothing, stdParsNull)
        {return base.setSmoothing(smoothing, stdPassThru);}

    stdbool flushSmoothly(stdParsNull)
        {return base.flushSmoothly(stdPassThru);}

public:

    bool overlayIsSet = false;

private:

    OverlaySmoother& base;
    ProcessKit const kit;

};

//----------------------------------------------------------------

}
