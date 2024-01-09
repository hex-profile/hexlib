#include "gpuBaseConsole.h"

//================================================================
//
// GpuBaseConsoleSplitter::*
//
//================================================================

#define SPLITTER(code) \
    bool aOk = errorBlock(a.code); \
    bool bOk = errorBlock(b.code); \
    \
    require(aOk && bOk); \
    \
    returnTrue;

////

stdbool GpuBaseConsoleSplitter::clear(stdPars(Kit))
    {SPLITTER(clear(stdPassNc))}

stdbool GpuBaseConsoleSplitter::update(stdPars(Kit))
    {SPLITTER(update(stdPassNc))}

stdbool GpuBaseConsoleSplitter::addImageBgr(const GpuMatrixAP<const uint8_x4>& img, const ImgOutputHint& hint, stdPars(Kit))
    {SPLITTER(addImageBgr(img, hint, stdPassNc))}

stdbool GpuBaseConsoleSplitter::overlayClear(stdPars(Kit))
    {SPLITTER(overlayClear(stdPassNc))}

stdbool GpuBaseConsoleSplitter::overlaySetImageBgr(const Point<Space>& size, const GpuImageProviderBgr32& img, const ImgOutputHint& hint, stdPars(Kit))
    {SPLITTER(overlaySetImageBgr(size, img, hint, stdPassNc))}

stdbool GpuBaseConsoleSplitter::overlaySetImageFake(stdPars(Kit))
    {SPLITTER(overlaySetImageFake(stdPassNc))}

stdbool GpuBaseConsoleSplitter::overlayUpdate(stdPars(Kit))
    {SPLITTER(overlayUpdate(stdPassNc))}

bool GpuBaseConsoleSplitter::getTextEnabled()
    {return a.getTextEnabled();}

void GpuBaseConsoleSplitter::setTextEnabled(bool textEnabled)
    {a.setTextEnabled(textEnabled); b.setTextEnabled(textEnabled);}

////

#undef SPLITTER
