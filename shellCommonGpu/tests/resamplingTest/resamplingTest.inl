namespace PREP_PASTE(FUNCNAME, PIXEL) {

//================================================================
//
// downsampleModelHorizontal
//
//================================================================

GPUTOOL_2D_BEG
(
    PREP_PASTE4(FUNCNAME, Horizontal, PIXEL, KERNEL),
    ((const PIXEL, src, INTERP_NONE, BORDER_MIRROR)),
    ((PIXEL, dst)),
    ((float32, downsampleFactor))
    ((KERNEL, filterKernel))
)
#if DEVCODE
{
    typedef VECTOR_REBASE_(PIXEL, float32) FloatType;
    FloatType sumWV = convertNearest<FloatType>(0);
    float32 sumW = 0;

    const float32 radius = clampMin(downsampleFactor, 1.f) * filterKernel.radius();
    float32 srcCenter = downsampleFactor * Xs;

    float32 srcOrg = srcCenter - radius; // space
    float32 srcEnd = srcCenter + radius;

    Space srcOrgGrid = convertDown<Space>(srcOrg - 0.5f);
    Space srcEndGrid = convertUp<Space>(srcEnd - 0.5f);

    for (Space iX = srcOrgGrid - 4; iX <= srcEndGrid + 4; ++iX)
    {
        float32 srcX = iX + 0.5f;
        auto value = tex2D(srcSampler, point(srcX, Ys) * srcTexstep);

        float32 w = filterKernel.func((srcX - srcCenter) / clampMin(downsampleFactor, 1.f)) / clampMin(downsampleFactor, 1.f);
        sumWV += w * value;
        sumW += w;
    }

    float32 divSumW = 1.f / sumW;
    if (sumW == 0) divSumW = 0;
    FloatType result = sumWV * divSumW;

    storeNorm(dst, result);
}
#endif
GPUTOOL_2D_END

//================================================================
//
// downsampleModelVertical
//
//================================================================

GPUTOOL_2D_BEG
(
    PREP_PASTE4(FUNCNAME, Vertical, PIXEL, KERNEL),
    ((const PIXEL, src, INTERP_NONE, BORDER_MIRROR)),
    ((PIXEL, dst)),
    ((float32, downsampleFactor))
    ((KERNEL, filterKernel))
)
#if DEVCODE
{
    typedef VECTOR_REBASE_(PIXEL, float32) FloatType;
    FloatType sumWV = convertNearest<FloatType>(0);
    float32 sumW = 0;

    const float32 radius = clampMin(downsampleFactor, 1.f) * filterKernel.radius();

    float32 srcCenter = downsampleFactor * Ys;

    float32 srcOrg = srcCenter - radius; // space
    float32 srcEnd = srcCenter + radius;

    Space srcOrgGrid = convertDown<Space>(srcOrg - 0.5f);
    Space srcEndGrid = convertUp<Space>(srcEnd - 0.5f);

    for (Space iY = srcOrgGrid; iY <= srcEndGrid; ++iY)
    {
        float32 srcY = iY + 0.5f;
        auto value = tex2D(srcSampler, point(Xs, srcY) * srcTexstep);

        float32 w = filterKernel.func((srcY - srcCenter) / clampMin(downsampleFactor, 1.f)) / clampMin(downsampleFactor, 1.f);
        sumWV += w * value;
        sumW += w;
    }

    float32 divSumW = 1.f/sumW;
    if (sumW == 0) divSumW = 0;
    FloatType result = sumWV * divSumW;

    storeNorm(dst, result);
}
#endif
GPUTOOL_2D_END

//================================================================
//
// downsampleModel
//
//================================================================

#if HOSTCODE

void FUNCNAME
(
    const GpuMatrix<const PIXEL>& src,
    const GpuMatrix<PIXEL>& dst,
    const Point<float32>& downsampleFactor,
    const KERNEL& filterKernel,
    stdPars(GpuProcessKit)
)
{
    GPU_MATRIX_ALLOC(tmp, PIXEL, point(dst.sizeX(), src.sizeY()));

    PREP_PASTE4(FUNCNAME, Horizontal, PIXEL, KERNEL)(src, tmp, downsampleFactor.X, filterKernel, stdPass);
    PREP_PASTE4(FUNCNAME, Vertical, PIXEL, KERNEL)(tmp, dst, downsampleFactor.Y, filterKernel, stdPass);
}

#endif

//----------------------------------------------------------------

}

HOST_ONLY(using PREP_PASTE(FUNCNAME, PIXEL)::FUNCNAME;)
