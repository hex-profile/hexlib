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

devDefineSampler(srcSampler1, DevSampler2D, DevSamplerFloat, 1)
devDefineSampler(srcSampler2, DevSampler2D, DevSamplerFloat, 2)

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

struct FilterCubic
{
    // Lanczos2
    static sysinline float32 C0() {return -0.00886333f;}
    static sysinline float32 C1() {return -0.04194003f;}
    static sysinline float32 C2() {return +0.11650009f;}
    static sysinline float32 C3() {return +0.43430327f;}
    static sysinline float32 C4() {return +0.43430327f;}
    static sysinline float32 C5() {return +0.11650009f;}
    static sysinline float32 C6() {return -0.04194003f;}
    static sysinline float32 C7() {return -0.00886333f;}
};

struct FilterGauss
{
    static sysinline float32 C0() {return 0.00814629f;}
    static sysinline float32 C1() {return 0.04819912f;}
    static sysinline float32 C2() {return 0.15767343f;}
    static sysinline float32 C3() {return 0.28517944f;}
    static sysinline float32 C4() {return 0.28517944f;}
    static sysinline float32 C5() {return 0.15767343f;}
    static sysinline float32 C6() {return 0.04819912f;}
    static sysinline float32 C7() {return 0.00814629f;}
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

//================================================================
//
// Kernel instantiations
//
//================================================================

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)) ((typename, FilterX)) ((typename, FilterY)), downsampleTwiceKernel_x1, DownsampleParams, downsampleTwiceKernelLink,
    downsampleTwiceCubic8u, (uint8) (FilterCubic) (FilterCubic));

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)) ((typename, FilterX)) ((typename, FilterY)), downsampleTwiceKernel_x1, DownsampleParams, downsampleTwiceKernelLink,
    downsampleTwiceGauss8u, (uint8) (FilterGauss) (FilterGauss));

//----------------------------------------------------------------

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)) ((typename, FilterX)) ((typename, FilterY)), downsampleTwiceKernel_x2, DownsampleParams, downsampleTwiceKernelLink,
    downsampleTwiceCubic8s_x2, (int8_x2) (FilterCubic) (FilterCubic));

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)) ((typename, FilterX)) ((typename, FilterY)), downsampleTwiceKernel_x2, DownsampleParams, downsampleTwiceKernelLink,
    downsampleTwiceGauss8s_x2, (int8_x2) (FilterGauss) (FilterGauss));

//----------------------------------------------------------------

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)) ((typename, FilterX)) ((typename, FilterY)), downsampleTwiceKernel_x1, DownsampleParams, downsampleTwiceKernelLink,
    downsampleTwiceGauss16f, (float16) (FilterGauss) (FilterGauss));

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)) ((typename, FilterX)) ((typename, FilterY)), downsampleTwiceKernel_x1, DownsampleParams, downsampleTwiceKernelLink,
    downsampleTwiceCubic16f, (float16) (FilterCubic) (FilterCubic));

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)) ((typename, FilterX)) ((typename, FilterY)), downsampleTwiceKernel_x1, DownsampleParams, downsampleTwiceKernelLink,
    downsampleTwiceCubicGauss16f, (float16) (FilterCubic) (FilterGauss));

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)) ((typename, FilterX)) ((typename, FilterY)), downsampleTwiceKernel_x1, DownsampleParams, downsampleTwiceKernelLink,
    downsampleTwiceGaussCubic16f, (float16) (FilterGauss) (FilterCubic));

//================================================================
//
// downsampleTwiceCubic
//
//================================================================

#if HOSTCODE

template <typename Src, typename Dst, typename FilterX, typename FilterY>
bool downsampleTwiceCubic(const GpuMatrix<const Src>& src, const GpuMatrix<Dst>& dst, const Point<Space>& srcOfs, stdPars(GpuProcessKit))
{
    stdBegin;

    if_not (kit.dataProcessing)
        return true;

    ////

    const int srcRank = VectorTypeRank<Src>::val;
    const GpuSamplerLink* srcSampler = (srcRank == 1) ? soft_cast<const GpuSamplerLink*>(&srcSampler1) : &srcSampler2;
    require(kit.gpuSamplerSetting.setSamplerImage(*srcSampler, src, BORDER_CLAMP, false, true, false, stdPass));

    ////

    require
    (
        kit.gpuKernelCalling.callKernel
        (
            divUpNonneg(dst.size(), point(threadCountX, threadCountY)),
            point(threadCountX, threadCountY),
            areaOf(dst),
            downsampleTwiceKernelLink<Dst, FilterX, FilterY>(),
            DownsampleParams<Dst, FilterX, FilterY>{srcOfs, dst},
            kit.gpuCurrentStream,
            stdPass
        )
    );

    ////

    stdEnd;
}

//----------------------------------------------------------------

INSTANTIATE_FUNC((downsampleTwiceCubic<uint8, uint8, FilterCubic, FilterCubic>))
INSTANTIATE_FUNC((downsampleTwiceCubic<uint8, uint8, FilterGauss, FilterGauss>))

INSTANTIATE_FUNC((downsampleTwiceCubic<int8_x2, int8_x2, FilterCubic, FilterCubic>))
INSTANTIATE_FUNC((downsampleTwiceCubic<int8_x2, int8_x2, FilterGauss, FilterGauss>))

INSTANTIATE_FUNC((downsampleTwiceCubic<float16, float16, FilterCubic, FilterCubic>))
INSTANTIATE_FUNC((downsampleTwiceCubic<float16, float16, FilterGauss, FilterGauss>))
INSTANTIATE_FUNC((downsampleTwiceCubic<float16, float16, FilterGauss, FilterCubic>))
INSTANTIATE_FUNC((downsampleTwiceCubic<float16, float16, FilterCubic, FilterGauss>))

//----------------------------------------------------------------

#endif
