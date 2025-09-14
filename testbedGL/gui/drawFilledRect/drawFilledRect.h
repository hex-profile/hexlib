#pragma once

#include "gpuProcessHeader.h"

//================================================================
//
// DrawSingleRectArgs
//
//================================================================

struct DrawSingleRectArgs
{
    Point<Space> org;
    Point<Space> end;
    uint8_x4 color;
};

//================================================================
//
// drawSingleRect
//
//================================================================

void drawSingleRect(const DrawSingleRectArgs& args, const GpuMatrix<uint8_x4>& dst, stdPars(GpuProcessKit));

//================================================================
//
// DrawFilledRectArgs
//
//================================================================

struct DrawFilledRectArgs
{
    DrawSingleRectArgs inner;
    DrawSingleRectArgs outer;
};

//================================================================
//
// drawFilledRect
//
//================================================================

void drawFilledRect(const DrawFilledRectArgs& args, const GpuMatrix<uint8_x4>& dst, stdPars(GpuProcessKit));
