#if HOSTCODE
#include "downsampleTwiceCubic.h"
#include "gpuAppliedApi/gpuAppliedApi.h"
#include "numbers/divRound.h"
#include "errorLog/errorLog.h"
#endif

#include "gpuDevice/gpuDevice.h"
#include "gpuDevice/loadstore/storeNorm.h"
#include "kit/kit.h"
#include "data/gpuMatrix.h"
#include "gpuSupport/gpuTemplateKernel.h"
#include "vectorTypes/vectorOperations.h"

//================================================================
//
// Downsampling 2X in SPACE coordinates.
//
// Compute kth dst element.
// Xk = k + 0.5 ; to space coord
// Xi = 2 Xk = 2k + 1 ; space in src
// I = Xi - 0.5 = 2k + 0.5
//
// Filter kernel continuous support is [-2, +2] in dst space,
// in src space it will be [-4, +4].
//
// Iminf = 2k + 0.5 - 4
// Imaxf = 2k + 0.5 + 4
//
// Compute range of covered integer indices:
// Imin = ceil(2k + 0.5 - 4) = 2k - 4 + ceil(0.5) = 2k - 4 + 1 = 2k - 3
// Imax = floor(2k + 0.5 + 4) = 2k + 4 + floor(0.5) = 2k + 4
//
// Kernel size is Imax-Imin+1 = 8 elements.
// First element is positioned at 2k - 3 src element for kth dst element.
//
//================================================================

//================================================================
//
// threadCountX
// threadCountY
//
// Each thread computes one destination pixel.
// So destination tile is (threadCountX, threadCountY)
//
//================================================================

static const Space threadCountX = 32;
static const Space threadCountY = 8;

//================================================================
//
// srcSampler
//
//================================================================

#if DEVCODE

devDefineSampler(srcSampler1, DevSampler2D, DevSamplerFloat, 1)
devDefineSampler(srcSampler2, DevSampler2D, DevSamplerFloat, 2)
devDefineSampler(srcSampler4, DevSampler2D, DevSamplerFloat, 4)

#endif

//================================================================
//
// DownsampleParams
//
//================================================================

template <typename Dst, typename FilterX, typename FilterY>
struct DownsampleParams
{
    Point<Space> srcOfs;
    GpuMatrix<Dst> dst;
};

//================================================================
//
// Filter coeffs (8 taps == 4 tap cubic * downsample 2X)
//
//================================================================

struct FilterLanczos2
{
    static sysinline float32 C0() {return -0.00886333f;}
    static sysinline float32 C1() {return -0.04194003f;}
    static sysinline float32 C2() {return +0.11650009f;}
    static sysinline float32 C3() {return +0.43430327f;}
    static sysinline float32 C4() {return +0.43430327f;}
    static sysinline float32 C5() {return +0.11650009f;}
    static sysinline float32 C6() {return -0.04194003f;}
    static sysinline float32 C7() {return -0.00886333f;}
};

//================================================================
//
// Instances
//
//================================================================

#define RANK 1
# include "downsampleTwiceCubic.inl"
#undef RANK

#define RANK 2
# include "downsampleTwiceCubic.inl"
#undef RANK

#define RANK 4
# include "downsampleTwiceCubic.inl"
#undef RANK

//================================================================
//
// Kernel instantiations
//
//================================================================

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)) ((typename, FilterX)) ((typename, FilterY)), downsampleTwiceKernel_x1, DownsampleParams, downsampleTwiceKernelLink,
    downsampleTwiceCubic8s_x1, (int8) (FilterLanczos2) (FilterLanczos2))

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)) ((typename, FilterX)) ((typename, FilterY)), downsampleTwiceKernel_x1, DownsampleParams, downsampleTwiceKernelLink,
    downsampleTwiceCubic8u_x1, (uint8) (FilterLanczos2) (FilterLanczos2))

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)) ((typename, FilterX)) ((typename, FilterY)), downsampleTwiceKernel_x1, DownsampleParams, downsampleTwiceKernelLink,
    downsampleTwiceCubic16s_x1, (int16) (FilterLanczos2) (FilterLanczos2))

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)) ((typename, FilterX)) ((typename, FilterY)), downsampleTwiceKernel_x1, DownsampleParams, downsampleTwiceKernelLink,
    downsampleTwiceCubic16u_x1, (uint16) (FilterLanczos2) (FilterLanczos2))

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)) ((typename, FilterX)) ((typename, FilterY)), downsampleTwiceKernel_x1, DownsampleParams, downsampleTwiceKernelLink,
    downsampleTwiceCubic16f_x1, (float16) (FilterLanczos2) (FilterLanczos2))

//----------------------------------------------------------------

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)) ((typename, FilterX)) ((typename, FilterY)), downsampleTwiceKernel_x2, DownsampleParams, downsampleTwiceKernelLink,
    downsampleTwiceCubic8s_x2, (int8_x2) (FilterLanczos2) (FilterLanczos2))

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)) ((typename, FilterX)) ((typename, FilterY)), downsampleTwiceKernel_x2, DownsampleParams, downsampleTwiceKernelLink,
    downsampleTwiceCubic8u_x2, (uint8_x2) (FilterLanczos2) (FilterLanczos2))

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)) ((typename, FilterX)) ((typename, FilterY)), downsampleTwiceKernel_x2, DownsampleParams, downsampleTwiceKernelLink,
    downsampleTwiceCubic16s_x2, (int16_x2) (FilterLanczos2) (FilterLanczos2))

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)) ((typename, FilterX)) ((typename, FilterY)), downsampleTwiceKernel_x2, DownsampleParams, downsampleTwiceKernelLink,
    downsampleTwiceCubic16u_x2, (uint16_x2) (FilterLanczos2) (FilterLanczos2))

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)) ((typename, FilterX)) ((typename, FilterY)), downsampleTwiceKernel_x2, DownsampleParams, downsampleTwiceKernelLink,
    downsampleTwiceCubic16f_x2, (float16_x2) (FilterLanczos2) (FilterLanczos2))

//----------------------------------------------------------------

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)) ((typename, FilterX)) ((typename, FilterY)), downsampleTwiceKernel_x4, DownsampleParams, downsampleTwiceKernelLink,
    downsampleTwiceCubic8s_x4, (int8_x4) (FilterLanczos2) (FilterLanczos2))

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)) ((typename, FilterX)) ((typename, FilterY)), downsampleTwiceKernel_x4, DownsampleParams, downsampleTwiceKernelLink,
    downsampleTwiceCubic8u_x4, (uint8_x4) (FilterLanczos2) (FilterLanczos2))

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)) ((typename, FilterX)) ((typename, FilterY)), downsampleTwiceKernel_x4, DownsampleParams, downsampleTwiceKernelLink,
    downsampleTwiceCubic16s_x4, (int16_x4) (FilterLanczos2) (FilterLanczos2))

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)) ((typename, FilterX)) ((typename, FilterY)), downsampleTwiceKernel_x4, DownsampleParams, downsampleTwiceKernelLink,
    downsampleTwiceCubic16u_x4, (uint16_x4) (FilterLanczos2) (FilterLanczos2))

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)) ((typename, FilterX)) ((typename, FilterY)), downsampleTwiceKernel_x4, DownsampleParams, downsampleTwiceKernelLink,
    downsampleTwiceCubic16f_x4, (float16_x4) (FilterLanczos2) (FilterLanczos2))

//================================================================
//
// downsampleTwiceCubic
//
//================================================================

#if HOSTCODE

template <typename Src, typename Dst>
void downsampleTwiceCubic(const GpuMatrix<const Src>& src, const GpuMatrix<Dst>& dst, const Point<Space>& srcOfs, stdPars(GpuProcessKit))
{
    if_not (kit.dataProcessing)
        return;

    using FilterX = FilterLanczos2;
    using FilterY = FilterLanczos2;

    ////

    const int srcRank = VectorTypeRank<Src>::val;

    const GpuSamplerLink* srcSampler = nullptr;
    if (srcRank == 1) srcSampler = &srcSampler1;
    if (srcRank == 2) srcSampler = &srcSampler2;
    if (srcRank == 4) srcSampler = &srcSampler4;
    REQUIRE(srcSampler);

    kit.gpuSamplerSetting.setSamplerImage(*srcSampler, src, BORDER_CLAMP, LinearInterpolation{false}, ReadNormalizedFloat{true}, NormalizedCoords{false}, stdPass);

    ////

    kit.gpuKernelCalling.callKernel
    (
        divUpNonneg(dst.size(), point(threadCountX, threadCountY)),
        point(threadCountX, threadCountY),
        areaOf(dst),
        downsampleTwiceKernelLink<Dst, FilterX, FilterY>(),
        DownsampleParams<Dst, FilterX, FilterY>{srcOfs, dst},
        kit.gpuCurrentStream,
        stdPass
    );
}

//----------------------------------------------------------------

INSTANTIATE_FUNC((downsampleTwiceCubic<int8, int8>))
INSTANTIATE_FUNC((downsampleTwiceCubic<uint8, uint8>))
INSTANTIATE_FUNC((downsampleTwiceCubic<int16, int8>))
INSTANTIATE_FUNC((downsampleTwiceCubic<uint16, uint8>))
INSTANTIATE_FUNC((downsampleTwiceCubic<float16, float16>))

INSTANTIATE_FUNC((downsampleTwiceCubic<int8_x2, int8_x2>))
INSTANTIATE_FUNC((downsampleTwiceCubic<uint8_x2, uint8_x2>))
INSTANTIATE_FUNC((downsampleTwiceCubic<int16_x2, int8_x2>))
INSTANTIATE_FUNC((downsampleTwiceCubic<uint16_x2, uint8_x2>))
INSTANTIATE_FUNC((downsampleTwiceCubic<float16_x2, float16_x2>))

INSTANTIATE_FUNC((downsampleTwiceCubic<int8_x4, int8_x4>))
INSTANTIATE_FUNC((downsampleTwiceCubic<uint8_x4, uint8_x4>))
INSTANTIATE_FUNC((downsampleTwiceCubic<int16_x4, int8_x4>))
INSTANTIATE_FUNC((downsampleTwiceCubic<uint16_x4, uint8_x4>))
INSTANTIATE_FUNC((downsampleTwiceCubic<float16_x4, float16_x4>))

//----------------------------------------------------------------

#endif
