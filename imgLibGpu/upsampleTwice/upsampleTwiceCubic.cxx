#if HOSTCODE
#include "upsampleTwiceCubic.h"
#include "gpuAppliedApi/gpuAppliedApi.h"
#include "numbers/divRound.h"
#include "errorLog/errorLog.h"
#include "dataAlloc/gpuMatrixMemory.h"
#endif

#include "kit/kit.h"
#include "data/gpuMatrix.h"
#include "gpuDevice/gpuDevice.h"
#include "gpuDevice/loadstore/storeNorm.h"
#include "readBordered.h"
#include "gpuSupport/gpuTemplateKernel.h"
#include "vectorTypes/vectorOperations.h"

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
// Filter coeffs
//
//================================================================

#if 0
#define C0 (-3 / 128.f)
#define C1 (-9 / 128.f)
#define C2 (+29 / 128.f)
#define C3 (+111 / 128.f)
#define C4 (+111 / 128.f)
#define C5 (+29 / 128.f)
#define C6 (-9 / 128.f)
#define C7 (-3 / 128.f)
#endif

#if 1 // Lanczos2
#define C0 (-0.01772666f)
#define C1 (-0.08388007f)
#define C2 (+0.23300019f)
#define C3 (+0.86860654f)
#define C4 (+0.86860654f)
#define C5 (+0.23300019f)
#define C6 (-0.08388007f)
#define C7 (-0.01772666f)
#endif

//================================================================
//
// UpsampleParams
//
//================================================================

template <typename Dst>
KIT_CREATE1_(UpsampleParams, GpuMatrix<Dst>, dst);

//================================================================
//
// srcSampler
//
//================================================================

devDefineSampler(srcSampler1, DevSampler2D, DevSamplerFloat, 1)
devDefineSampler(srcSampler2, DevSampler2D, DevSamplerFloat, 2)

//================================================================
//
// Kernel instances
//
//================================================================

#define RANK 1
#define DST uint8
# include "upsampleTwiceCubicKernel.inl"
#undef RANK
#undef DST

#define RANK 1
#define DST float16
# include "upsampleTwiceCubicKernel.inl"
#undef RANK
#undef DST

#define RANK 2
#define DST uint8_x2
# include "upsampleTwiceCubicKernel.inl"
#undef RANK
#undef DST

#define RANK 2
#define DST float16_x2
# include "upsampleTwiceCubicKernel.inl"
#undef RANK
#undef DST

//================================================================
//
// upsampleTwiceCubic
//
//================================================================

#if HOSTCODE

template <typename Src, typename Dst>
bool upsampleTwiceCubic(const GpuMatrix<const Src>& src, const GpuMatrix<Dst>& dst, stdPars(GpuProcessKit))
{
    stdBegin;

    if_not (kit.dataProcessing)
        return true;

    //
    // The kernel covers destination 2k+1, 2k+2.
    // Let dst index be 0..N-1.
    //
    // 2k+1 <= 0 | 2k <= -1 | k <= -1/2 | k <= -1
    // 2k+2 >= N-1 | 2k >= N-3 | k >= (N-3)/2 | k >= ceil((N-3)/2) == ceil((N+1)/2) - 2
    //
    // So, k range is [-1, ceil((N+1)/2) - 2], or [-1, ceil((N+1)/2) - 1)
    //

    Point<Space> srcOrg = point(-1);
    Point<Space> srcEnd = dst.size() >> 1;

    REQUIRE(2*srcOrg+1 <= 0);

    REQUIRE(2*(srcEnd-1)+2 >= dst.size()-1);
    REQUIRE(2*(srcEnd-2)+2 < dst.size()-1);

    Point<Space> srcRange = srcEnd - srcOrg;

    ////

    const int srcRank = VectorTypeRank<Src>::val;
    const GpuSamplerLink* srcSampler = (srcRank == 1) ? soft_cast<const GpuSamplerLink*>(&srcSampler1) : &srcSampler2;
    require(kit.gpuSamplerSetting.setSamplerImage(*srcSampler, src, BORDER_CLAMP, false, true, false, stdPass));

    ////

    require
    (
        kit.gpuKernelCalling.callKernel
        (
            divUpNonneg(srcRange, point(threadCountX, threadCountY)),
            point(threadCountX, threadCountY),
            areaOf(dst),
            upsampleTwiceKernelLink<Dst>(),
            UpsampleParams<Dst>(dst),
            kit.gpuCurrentStream,
            stdPass
        )
    );

    ////

    stdEnd;
}

//----------------------------------------------------------------

INSTANTIATE_FUNC((upsampleTwiceCubic<uint8, uint8>))
INSTANTIATE_FUNC((upsampleTwiceCubic<float16, float16>))
INSTANTIATE_FUNC((upsampleTwiceCubic<uint8_x2, uint8_x2>))
INSTANTIATE_FUNC((upsampleTwiceCubic<float16_x2, float16_x2>))

//----------------------------------------------------------------

#endif
