#pragma once

#include "gpuProcessHeader.h"

//================================================================
//
// convertBgr32ToMono
// convertMonoToBgr32
//
//================================================================

stdbool convertBgr32ToMono(const GpuMatrixAP<const uint8_x4>& src, const GpuMatrixAP<uint8>& dst, stdPars(GpuProcessKit));
stdbool convertMonoToBgr32(const GpuMatrixAP<const uint8>& src, const GpuMatrixAP<uint8_x4>& dst, stdPars(GpuProcessKit));

stdbool convertBgr32ToMonoBgr32(const GpuMatrixAP<const uint8_x4>& src, const GpuMatrixAP<uint8_x4>& dst, stdPars(GpuProcessKit));

//================================================================
//
// convertBgr32ToBgr24
// convertBgr24ToBgr32
//
//================================================================

stdbool convertBgr32ToBgr24(const GpuMatrixAP<const uint8_x4>& src, const GpuMatrixAP<uint8>& dst, stdPars(GpuProcessKit));
stdbool convertBgr24ToBgr32(const GpuMatrixAP<const uint8>& src, const GpuMatrixAP<uint8_x4>& dst, stdPars(GpuProcessKit));

//================================================================
//
// convertBgr24ToMono
//
//================================================================

stdbool convertBgr24ToMono(const GpuMatrixAP<const uint8>& src, const GpuMatrixAP<uint8>& dst, stdPars(GpuProcessKit));
