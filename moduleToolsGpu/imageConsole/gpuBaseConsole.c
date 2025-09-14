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
    require(aOk && bOk);

////

void GpuBaseConsoleSplitter::clear(stdPars(Kit))
    {SPLITTER(clear(stdPassNc))}

void GpuBaseConsoleSplitter::update(stdPars(Kit))
    {SPLITTER(update(stdPassNc))}

void GpuBaseConsoleSplitter::addImageBgr(const GpuMatrixAP<const uint8_x4>& img, const ImgOutputHint& hint, stdPars(Kit))
    {SPLITTER(addImageBgr(img, hint, stdPassNc))}

void GpuBaseConsoleSplitter::overlayClear(stdPars(Kit))
    {SPLITTER(overlayClear(stdPassNc))}

void GpuBaseConsoleSplitter::overlaySetImageBgr(const Point<Space>& size, const GpuImageProviderBgr32& img, const ImgOutputHint& hint, stdPars(Kit))
    {SPLITTER(overlaySetImageBgr(size, img, hint, stdPassNc))}

void GpuBaseConsoleSplitter::overlaySetImageFake(stdPars(Kit))
    {SPLITTER(overlaySetImageFake(stdPassNc))}

void GpuBaseConsoleSplitter::overlayUpdate(stdPars(Kit))
    {SPLITTER(overlayUpdate(stdPassNc))}

bool GpuBaseConsoleSplitter::getTextEnabled()
    {return a.getTextEnabled();}

void GpuBaseConsoleSplitter::setTextEnabled(bool textEnabled)
    {a.setTextEnabled(textEnabled); b.setTextEnabled(textEnabled);}

////

#undef SPLITTER
