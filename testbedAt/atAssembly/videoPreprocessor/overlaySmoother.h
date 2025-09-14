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

    void init(stdPars(InitKit));
    void deinit();

public:

    void setImage(const Point<Space>& size, BaseImageProvider& imageProvider, const FormatOutputAtom& desc, uint32 id, bool textEnabled, stdPars(ProcessKit));
    void setImageFake(stdPars(ProcessKit)) {}
    void updateImage(stdPars(ProcessKit));
    void clearQueue(stdPars(ProcessKit));
    void setSmoothing(bool smoothing, stdPars(ProcessKit));
    void flushSmoothly(stdPars(ProcessKit));

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

    void setImage(const Point<Space>& size, bool dataProcessing, BaseImageProvider& imageProvider, const FormatOutputAtom& desc, uint32 id, bool textEnabled, stdParsNull)
        {overlayIsSet = true; base.setImage(size, imageProvider, desc, id, textEnabled, stdPassThru);}

    void setImageFake(stdParsNull)
        {overlayIsSet = true; base.setImageFake(stdPassThru);}

    void updateImage(stdParsNull)
        {base.updateImage(stdPassThru);}

    void clearQueue(stdParsNull)
        {base.clearQueue(stdPassThru);}

    void setSmoothing(bool smoothing, stdParsNull)
        {base.setSmoothing(smoothing, stdPassThru);}

    void flushSmoothly(stdParsNull)
        {base.flushSmoothly(stdPassThru);}

public:

    bool overlayIsSet = false;

private:

    OverlaySmoother& base;
    ProcessKit const kit;

};

//----------------------------------------------------------------

}
