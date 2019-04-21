#pragma once

#include "gpuProcessHeader.h"

//================================================================
//
// Filters
//
//================================================================

struct FilterCubic;
struct FilterGauss;

//================================================================
//
// downsampleTwiceCubic
//
// 0.115 ms FullHD
//
//================================================================

template <typename Src, typename Dst, typename FilterX, typename FilterY>
stdbool downsampleTwiceCubic(const GpuMatrix<const Src>& src, const GpuMatrix<Dst>& dst, const Point<Space>& srcOfs, stdPars(GpuProcessKit));
