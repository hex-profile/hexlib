//================================================================
//
// Input parameters.
//
//================================================================

#if !defined(MASK_ENABLED)
    #error
#endif

//----------------------------------------------------------------

#if !(defined(INPUT_PIXEL) && defined(COMPLEX_PIXEL))
    #error Type parameters are required.
#endif

//----------------------------------------------------------------

#if !(defined(FUNCNAME) && defined(GABOR_BANK))
    #error
#endif

//----------------------------------------------------------------

#if !(defined(ORIENT_COUNT) && defined(COMPRESS_OCTAVES) && defined(INPUT_BORDER_MODE))
    #error
#endif

//----------------------------------------------------------------

#undef DIR

#if !defined(HORIZONTAL)
    #error HORIZONTAL should be defined to 0 or 1
#elif HORIZONTAL
    #define DIR(h, v) h
#else
    #define DIR(h, v) v
#endif

//----------------------------------------------------------------

#if !(defined(POSTPROCESS_PARAMS) && defined(POSTPROCESS_ACTION))
    #error
#endif

//----------------------------------------------------------------

#if MASK_ENABLED
    #if !(defined(MASK_PIXEL) && defined(MASK_PARAMS) && defined(MASK_CHECK) && defined(MASK_BORDER_MODE) && defined(DEFAULT_WEIGHTED_PIXEL) && defined(DEFAULT_PIXEL))
        #error
    #endif
#endif

//================================================================
//
// GABOR_FINAL_FACTOR
//
//================================================================

#define GABOR_FINAL_FACTOR 4.f

//================================================================
//
// GABOR_INITIAL_CACHED_THREAD_COUNT
// GABOR_FINAL_CACHED_THREAD_COUNT
//
//================================================================

#define GABOR_INITIAL_CACHED_THREAD_COUNT \
    DIR((64, 4), (32, 8))

#define GABOR_FINAL_CACHED_THREAD_COUNT \
    DIR((32, 4), (32, 8))

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Gabor processing.
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================


//================================================================
//
// gaborProcessInitialSimple
//
//================================================================

GPUTOOL_2D_BEG
(
    PREP_PASTE3(FUNCNAME, ProcessInitialSimple, DIR(Hor, Ver)),
    ((const INPUT_PIXEL, src, INTERP_NEAREST, INPUT_BORDER_MODE))
    ((const float32_x2, circleTable, INTERP_LINEAR, BORDER_WRAP)),
    GPUTOOL_INDEXED_NAME(ORIENT_COUNT, COMPLEX_PIXEL, dst),
    ((bool, demodulateOutput))
)
#if DEVCODE
{

    constexpr Space filterSize = PREP_PASTE(GABOR_BANK, DIR(SizeX, SizeY));
    COMPILE_ASSERT(!(COMPRESS_OCTAVES == 0) || (filterSize % 2 == 1));
    COMPILE_ASSERT(!(COMPRESS_OCTAVES >= 1) || (filterSize % 2 == 0));

    //----------------------------------------------------------------
    //
    // Map to source image.
    //
    //----------------------------------------------------------------

    Point<Space> dstIdx = point(X, Y);
    constexpr Space downsampleFactor = (1 << COMPRESS_OCTAVES);
    Point<Space> srcIdx = dstIdx;
    srcIdx.DIR(X, Y) = mapDownsampleIndexToSource<downsampleFactor, filterSize>(dstIdx.DIR(X, Y));

    //----------------------------------------------------------------
    //
    // Loop.
    //
    //----------------------------------------------------------------

    #define TMP_MACRO(k, _) \
        float32_x2 sum##k = make_float32_x2(0, 0);

    PREP_FOR(ORIENT_COUNT, TMP_MACRO, _)

    #undef TMP_MACRO

    ////

    Point<float32> srcReadTexPos = convertIndexToPos(srcIdx) * srcTexstep;

    ////

    devUnrollLoop
    for (Space i = 0; i < filterSize; ++i)
    {
        Point<Space> ofs = point(0);
        ofs.DIR(X, Y) = i;

        float32 value = tex2D(srcSampler, srcReadTexPos + convertFloat32(ofs) * srcTexstep);

        #define TMP_MACRO(k, _) \
            sum##k += PREP_PASTE3(GABOR_BANK, DIR(DataX, DataY), k)[i] * value;

        PREP_FOR(ORIENT_COUNT, TMP_MACRO, PREP_EMPTY)

        #undef TMP_MACRO
    }

    ////

    float32 filterCenter = DIR(Xs, Ys) * (1 << COMPRESS_OCTAVES);

    #define TMP_MACRO(k, _) \
        \
        if (demodulateOutput) \
            sum##k = complexMul(sum##k, devTex2D(circleTableSampler, -PREP_PASTE3(GABOR_BANK, Freq, k).DIR(x, y) * filterCenter, 0)); \
        \
        storeNorm(dst##k, sum##k);

    PREP_FOR(ORIENT_COUNT, TMP_MACRO, _)

    #undef TMP_MACRO

}
#endif
GPUTOOL_2D_END

//================================================================
//
// gaborProcessInitialCached
//
//================================================================

GPUTOOL_2D_BEG_EX
(
    PREP_PASTE3(FUNCNAME, ProcessInitialCached, DIR(Hor, Ver)),
    GABOR_INITIAL_CACHED_THREAD_COUNT,
    true,
    ((const INPUT_PIXEL, src, INTERP_NEAREST, INPUT_BORDER_MODE))
    ((const float32_x2, circleTable, INTERP_LINEAR, BORDER_WRAP)),
    GPUTOOL_INDEXED_NAME(ORIENT_COUNT, COMPLEX_PIXEL, dst),
    ((bool, demodulateOutput))
)
#if DEVCODE
{

    constexpr Space filterSize = PREP_PASTE(GABOR_BANK, DIR(SizeX, SizeY));
    COMPILE_ASSERT(!(COMPRESS_OCTAVES == 0) || (filterSize % 2 == 1));
    COMPILE_ASSERT(!(COMPRESS_OCTAVES >= 1) || (filterSize % 2 == 0));

    //
    // Map tile origin to source image
    //

    Point<Space> dstBase = vTileOrg;
    Point<Space> srcBase = dstBase;

    constexpr Space downsampleFactor = (1 << COMPRESS_OCTAVES);
    srcBase.DIR(X, Y) = mapDownsampleIndexToSource<downsampleFactor, filterSize>(srcBase.DIR(X, Y));

    //
    // ith thread first access: srcBase + i * downsampleFactor
    // ith thread last access: srcBase + i * downsampleFactor + filterSize - 1
    //
    // first access: srcBase
    // last access: srcBase + (threadCount-1) * downsampleFactor + filterSize - 1
    //
    // cache size: (threadCount-1) * downsampleFactor + filterSize
    //

    constexpr Space threadCountX = vTileSizeX;
    constexpr Space threadCountY = vTileSizeY;
    Point<Space> threadIdx = vTileMember;

    constexpr Space cacheSizeX = DIR((threadCountX-1) * downsampleFactor + filterSize, threadCountX);
    constexpr Space cacheSizeY = DIR(threadCountY, (threadCountY-1) * downsampleFactor + filterSize);

    //----------------------------------------------------------------
    //
    // Load src block.
    //
    //----------------------------------------------------------------

    devSramMatrixFor2dAccess(cache, float32, cacheSizeX, cacheSizeY, threadCountX);

    MatrixPtr(float32) cacheLoadPtr = MATRIX_POINTER_(cache, threadIdx);

    ////

    Point<Space> srcReadIdx = srcBase + threadIdx;

    ////

    Point<float32> srcReadTexPos = convertIndexToPos(srcReadIdx) * srcTexstep;

    PARALLEL_LOOP_2D_UNBASED
    (
        iX, iY, cacheSizeX, cacheSizeY, threadIdx.X, threadIdx.Y, threadCountX, threadCountY,
        *(cacheLoadPtr + iX + iY * cacheMemPitch) = tex2D(srcSampler, srcReadTexPos + point(float32(iX), float32(iY)) * srcTexstep);
    )

    ////

    devSyncThreads();

    //----------------------------------------------------------------
    //
    // Exit if not producing output.
    //
    //----------------------------------------------------------------

    if_not (vItemIsActive)
        return;

    //----------------------------------------------------------------
    //
    // Loop.
    //
    //----------------------------------------------------------------

    #define TMP_MACRO(k, _) \
        float32_x2 sum##k = make_float32_x2(0, 0);

    PREP_FOR(ORIENT_COUNT, TMP_MACRO, _)

    #undef TMP_MACRO

    ////
  

    Point<Space> extendedIdx = threadIdx;
    extendedIdx.DIR(X, Y) *= downsampleFactor; // 2X bank conflicts, but not important.
    MatrixPtr(const float32) cachePtr = MATRIX_POINTER_(cache, extendedIdx);

    ////

    devUnrollLoop
    for (Space i = 0; i < filterSize; ++i)
    {
        float32 value = *cachePtr;
        cachePtr += DIR(1, cacheMemPitch);

        #define TMP_MACRO(k, _) \
            sum##k += PREP_PASTE3(GABOR_BANK, DIR(DataX, DataY), k)[i] * value;

        PREP_FOR(ORIENT_COUNT, TMP_MACRO, _)

        #undef TMP_MACRO
    }

    ////

    float32 filterCenter = DIR(Xs, Ys) * (1 << COMPRESS_OCTAVES);

    #define TMP_MACRO(k, _) \
        \
        if (demodulateOutput) \
            sum##k = complexMul(sum##k, devTex2D(circleTableSampler, -PREP_PASTE3(GABOR_BANK, Freq, k).DIR(x, y) * filterCenter, 0)); \
        \
        storeNorm(dst##k, sum##k);

    PREP_FOR(ORIENT_COUNT, TMP_MACRO, _)

    #undef TMP_MACRO

}
#endif
GPUTOOL_2D_END_EX

//================================================================
//
// gaborProcessFinalSimple
//
//================================================================

GPUTOOL_2D_BEG
(
    PREP_PASTE3(FUNCNAME, ProcessFinalSimple, DIR(Hor, Ver)),
    GPUTOOL_INDEXED_SAMPLER(ORIENT_COUNT, const COMPLEX_PIXEL, src, INTERP_NONE, INPUT_BORDER_MODE)
    ((const float32_x2, circleTable, INTERP_LINEAR, BORDER_WRAP)),
    GPUTOOL_INDEXED_NAME(ORIENT_COUNT, COMPLEX_PIXEL, dst),
    ((bool, demodulateOutput))
    ((POSTPROCESS_PARAMS, postprocessParams))
)
#if DEVCODE
{

    constexpr Space filterSize = PREP_PASTE(GABOR_BANK, DIR(SizeX, SizeY));
    COMPILE_ASSERT(!(COMPRESS_OCTAVES == 0) || (filterSize % 2 == 1));
    COMPILE_ASSERT(!(COMPRESS_OCTAVES >= 1) || (filterSize % 2 == 0));

    ////

    constexpr Space downsampleFactor = (1 << COMPRESS_OCTAVES);

    Point<Space> dstIdx = point(X, Y);
    Point<Space> srcIdx = dstIdx;
    srcIdx.DIR(X, Y) = mapDownsampleIndexToSource<downsampleFactor, filterSize>(dstIdx.DIR(X, Y));

    //----------------------------------------------------------------
    //
    // Make all orientations.
    //
    //----------------------------------------------------------------

    Point<float32> srcReadTexPos = convertIndexToPos(srcIdx) * src0Texstep;

    ////

    #define TMP_MACRO(k, _) \
        \
        float32_x2 sum##k = make_float32_x2(0, 0); \
        \
        devUnrollLoop \
        for (Space i = 0; i < filterSize; ++i) \
        { \
            Point<Space> ofs = point(0); \
            ofs.DIR(X, Y) = i; \
            \
            float32_x2 value = tex2D(src##k##Sampler, srcReadTexPos + convertFloat32(ofs) * src0Texstep); \
            sum##k = complexMad(sum##k, PREP_PASTE3(GABOR_BANK, DIR(DataX, DataY), k)[i], value); \
        }

    PREP_FOR(ORIENT_COUNT, TMP_MACRO, _)

    #undef TMP_MACRO

    //----------------------------------------------------------------
    //
    // Exit if not producing output value.
    //
    //----------------------------------------------------------------

    if_not (vItemIsActive)
        return;

    //----------------------------------------------------------------
    //
    // Demodulate (and apply output factor).
    //
    //----------------------------------------------------------------

    float32 filterCenter = DIR(Xs, Ys) * (1 << COMPRESS_OCTAVES);

    #define TMP_MACRO(k, _) \
        \
        if (demodulateOutput) \
            sum##k = complexMul(sum##k, devTex2D(circleTableSampler, -PREP_PASTE3(GABOR_BANK, Freq, k).DIR(x, y) * filterCenter, 0)); \
        \
        sum##k *= GABOR_FINAL_FACTOR;

    PREP_FOR(ORIENT_COUNT, TMP_MACRO, _)

    #undef TMP_MACRO

    //----------------------------------------------------------------
    //
    // Postprocess.
    //
    //----------------------------------------------------------------

    POSTPROCESS_ACTION(point(X, Y), vGlobSize, PREP_ENUM_INDEXED(ORIENT_COUNT, sum), postprocessParams);

    //----------------------------------------------------------------
    //
    // Store.
    //
    //----------------------------------------------------------------

    #define TMP_MACRO(k, _) \
        storeNorm(dst##k, sum##k);

    PREP_FOR(ORIENT_COUNT, TMP_MACRO, _)

    #undef TMP_MACRO

}
#endif
GPUTOOL_2D_END

//================================================================
//
// gaborProcessFinalCached
//
//================================================================

GPUTOOL_2D_BEG_EX
(
    PREP_PASTE3(FUNCNAME, ProcessFinalCached, DIR(Hor, Ver)),
    GABOR_FINAL_CACHED_THREAD_COUNT,
    true,
    GPUTOOL_INDEXED_SAMPLER(ORIENT_COUNT, const COMPLEX_PIXEL, src, INTERP_NONE, INPUT_BORDER_MODE)
    ((const float32_x2, circleTable, INTERP_LINEAR, BORDER_WRAP)),
    GPUTOOL_INDEXED_NAME(ORIENT_COUNT, COMPLEX_PIXEL, dst),
    ((bool, demodulateOutput))
    ((POSTPROCESS_PARAMS, postprocessParams))
)
#if DEVCODE
{

    constexpr Space filterSize = PREP_PASTE(GABOR_BANK, DIR(SizeX, SizeY));
    COMPILE_ASSERT(!(COMPRESS_OCTAVES == 0) || (filterSize % 2 == 1));
    COMPILE_ASSERT(!(COMPRESS_OCTAVES >= 1) || (filterSize % 2 == 0));

    ////

    constexpr Space downsampleFactor = (1 << COMPRESS_OCTAVES);

    Point<Space> dstBase = vTileOrg;
    Point<Space> srcBase = dstBase;
    srcBase.DIR(X, Y) = mapDownsampleIndexToSource<downsampleFactor, filterSize>(srcBase.DIR(X, Y));

    //
    // ith thread first access: srcBase + i * downsampleFactor
    // ith thread last access: srcBase + i * downsampleFactor + filterSize - 1
    //
    // first access: srcBase
    // last access: srcBase + (threadCount-1) * downsampleFactor + filterSize - 1
    //
    // cache size: (threadCount-1) * downsampleFactor + filterSize
    //

    constexpr Space threadCountX = vTileSizeX;
    constexpr Space threadCountY = vTileSizeY;
    Point<Space> threadIdx = vTileMember;

    constexpr Space cacheSizeX = DIR((threadCountX-1) * downsampleFactor + filterSize, threadCountX);
    constexpr Space cacheSizeY = DIR(threadCountY, (threadCountY-1) * downsampleFactor + filterSize);

    //----------------------------------------------------------------
    //
    // Load src block.
    //
    //----------------------------------------------------------------

    devSramMatrixFor2dAccess(cache, float32_x2, cacheSizeX, cacheSizeY, threadCountX);

    ////

    MatrixPtr(float32_x2) cacheLoadPtr = MATRIX_POINTER_(cache, threadIdx);
    Point<Space> srcReadIdx = srcBase + threadIdx;
    Point<float32> srcReadTexPos = convertIndexToPos(srcReadIdx) * src0Texstep;

    ////

    #define LOAD_SRC_BLOCK(sampler) \
        \
        PARALLEL_LOOP_2D_UNBASED \
        ( \
            iX, iY, cacheSizeX, cacheSizeY, threadIdx.X, threadIdx.Y, threadCountX, threadCountY, \
            *(cacheLoadPtr + iX + iY * cacheMemPitch) = tex2D(sampler, srcReadTexPos + point(float32(iX), float32(iY)) * src0Texstep); \
        )

    //----------------------------------------------------------------
    //
    // Make all orientations.
    //
    //----------------------------------------------------------------

    Point<Space> extendedIdx = threadIdx;
    extendedIdx.DIR(X, Y) *= downsampleFactor; // 2X bank conflicts, but not important.
    MatrixPtr(const float32_x2) cacheStartPtr = MATRIX_POINTER_(cache, extendedIdx);

    ////

    #define TMP_MACRO(k, _) \
        \
        float32_x2 sum##k = make_float32_x2(0, 0); \
        \
        /* Finish using cache */ \
        if (k != 0) devSyncThreads(); \
        \
        /* Load block */ \
        LOAD_SRC_BLOCK(src##k##Sampler) \
        devSyncThreads(); \
        \
        devUnrollLoop \
        for (Space i = 0; i < filterSize; ++i) \
        { \
            float32_x2 value = cacheStartPtr[i * DIR(1, cacheMemPitch)]; \
            sum##k = complexMad(sum##k, PREP_PASTE3(GABOR_BANK, DIR(DataX, DataY), k)[i], value); \
        }

    PREP_FOR(ORIENT_COUNT, TMP_MACRO, _)

    #undef TMP_MACRO

    //----------------------------------------------------------------
    //
    // Exit if not producing output value.
    //
    //----------------------------------------------------------------

    if_not (vItemIsActive)
        return;

    //----------------------------------------------------------------
    //
    // Demodulate (and apply output factor).
    //
    //----------------------------------------------------------------

    float32 filterCenter = DIR(Xs, Ys) * (1 << COMPRESS_OCTAVES);

    #define TMP_MACRO(k, _) \
        \
        if (demodulateOutput) \
            sum##k = complexMul(sum##k, devTex2D(circleTableSampler, -PREP_PASTE3(GABOR_BANK, Freq, k).DIR(x, y) * filterCenter, 0)); \
        \
        sum##k *= GABOR_FINAL_FACTOR;

    PREP_FOR(ORIENT_COUNT, TMP_MACRO, _)

    #undef TMP_MACRO

    //----------------------------------------------------------------
    //
    // Postprocess.
    //
    //----------------------------------------------------------------

    POSTPROCESS_ACTION(point(X, Y), vGlobSize, PREP_ENUM_INDEXED(ORIENT_COUNT, sum), postprocessParams);

    //----------------------------------------------------------------
    //
    // Store.
    //
    //----------------------------------------------------------------

    #define TMP_MACRO(k, _) \
        storeNorm(dst##k, sum##k);

    PREP_FOR(ORIENT_COUNT, TMP_MACRO, _)

    #undef TMP_MACRO

}
#endif
GPUTOOL_2D_END_EX

//================================================================
//
// gaborProcessFull
//
//================================================================

#if HOSTCODE

template <int=0>
stdbool PREP_PASTE3(FUNCNAME, ProcessFull, DIR(Hor, Ver))
(
    const GpuMatrix<const INPUT_PIXEL>& src, 
    const GpuMatrix<const float32_x2>& circleTable,
    const GpuLayeredMatrix<COMPLEX_PIXEL>& dst,
    bool demodulateOutput,
    const POSTPROCESS_PARAMS& postprocessParams,
    bool uncachedVersion,
    stdPars(GpuProcessKit)
)
{
    Point<Space> tmpSize = src.size();
    tmpSize.DIR(X, Y) = dst.size().DIR(X, Y);

    ////

    GPU_LAYERED_MATRIX_ALLOC(tmp, COMPLEX_PIXEL, ORIENT_COUNT, tmpSize);

    ////

    require
    (
        (
            uncachedVersion ?
                PREP_PASTE3(FUNCNAME, ProcessInitialSimple, DIR(Hor, Ver)) :
                PREP_PASTE3(FUNCNAME, ProcessInitialCached, DIR(Hor, Ver))
        )
        (
            src, 
            circleTable, 
            GPU_LAYERED_MATRIX_PASS(ORIENT_COUNT, tmp), 
            demodulateOutput,
            stdPass
        )
    );

    ////

    require
    (
        ( 
            uncachedVersion ? 
            PREP_PASTE3(FUNCNAME, ProcessFinalSimple, DIR(Ver, Hor)) :
            PREP_PASTE3(FUNCNAME, ProcessFinalCached, DIR(Ver, Hor))
        )
        (
            GPU_LAYERED_MATRIX_PASS(ORIENT_COUNT, tmp), 
            circleTable, 
            GPU_LAYERED_MATRIX_PASS(ORIENT_COUNT, dst), 
            demodulateOutput,
            postprocessParams,
            stdPass
        )
    );

    ////

    returnTrue;
}

#endif

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Gabor fixing.
//
// If some of input pixels are undefined, they cannot be
// simply fed into Gabor filters or replaced with zeros.
//
// The simplest way of handling undefined pixels is:
//
// * For each Gabor position, compute a default input value:
// 2D weighted average of defined pixels with weight window of
// Gabor envelope, Gaussian ball.
//
// * When computing each Gabor filter, replace undefined input pixels with 
// the default value. The replacement cannot be done inside the input image, 
// because the same undefined pixel may have different default values 
// when it is used for different Gabor positions.
//
// The above description prevents separable filtering if implemented directly.
// To keep filtering separable, the following method is used:
//
// Assume we don't know the default input value at the filtering stage, 
// let's denote it by variable D. 
//
// For an unconditional input, the filtered value is sum(Fi * Vi)
// where F is the filter and V is value.
//
// For conditional input, handled by replacing invalid pixels with value D,
// the input value is (Mi * Vi + (1 - Mi) * D), where M is mask (or probability).
//
// sum(Fi * (Mi * Vi + (1 - Mi) * D))
// ==
// sum(Fi * Mi * Vi) + D * sum(Fi * 1) - sum(Fi * Mi)
//
// So, the result of any linear filter, including complex Gabors, splits into three sums: 
// (Filtered product of image and mask) + (Filtered 1) - D * (Filtered mask).
//
// For Gabor filters, filtered 1 is zero, so the result is:
// (Gabor-filtered product of image and mask) - D * (Gabor-filtered mask).
//
// Both filters can be computed separably, as well as the default value D, which is 
// (envelope-filtered product of image and mask) / (envelope-filtered mask).
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

#if MASK_ENABLED

//================================================================
//
// gaborDefaultInitialCached
//
//================================================================

GPUTOOL_2D_BEG_EX
(
    PREP_PASTE3(FUNCNAME, DefaultInitialCached, DIR(Hor, Ver)),
    GABOR_INITIAL_CACHED_THREAD_COUNT,
    true,
    ((const MASK_PIXEL, mask, INTERP_NEAREST, MASK_BORDER_MODE))
    ((const INPUT_PIXEL, image, INTERP_NEAREST, INPUT_BORDER_MODE)),
    ((DEFAULT_WEIGHTED_PIXEL, dst)),
    ((MASK_PARAMS, maskParams))
)
#if DEVCODE
{
    constexpr Space filterSize = PREP_PASTE(GABOR_BANK, DIR(SizeX, SizeY));
    COMPILE_ASSERT(!(COMPRESS_OCTAVES == 0) || (filterSize % 2 == 1));
    COMPILE_ASSERT(!(COMPRESS_OCTAVES >= 1) || (filterSize % 2 == 0));

    //
    // Map tile origin to source image
    //

    Point<Space> dstBase = vTileOrg;
    Point<Space> srcBase = dstBase;

    constexpr Space downsampleFactor = (1 << COMPRESS_OCTAVES);
    srcBase.DIR(X, Y) = mapDownsampleIndexToSource<downsampleFactor, filterSize>(srcBase.DIR(X, Y));

    //
    // ith thread first access: srcBase + i * downsampleFactor
    // ith thread last access: srcBase + i * downsampleFactor + filterSize - 1
    //
    // first access: srcBase
    // last access: srcBase + (threadCount-1) * downsampleFactor + filterSize - 1
    //
    // cache size: (threadCount-1) * downsampleFactor + filterSize
    //

    constexpr Space threadCountX = vTileSizeX;
    constexpr Space threadCountY = vTileSizeY;
    Point<Space> threadIdx = vTileMember;

    constexpr Space cacheSizeX = DIR((threadCountX-1) * downsampleFactor + filterSize, threadCountX);
    constexpr Space cacheSizeY = DIR(threadCountY, (threadCountY-1) * downsampleFactor + filterSize);

    //----------------------------------------------------------------
    //
    // Load src block.
    //
    //----------------------------------------------------------------

    devSramMatrixFor2dAccess(maskCache, float32, cacheSizeX, cacheSizeY, threadCountX);
    auto maskCacheLoadPtr = MATRIX_POINTER_(maskCache, threadIdx);

    devSramMatrixFor2dAccess(prodCache, float32, cacheSizeX, cacheSizeY, threadCountX);
    auto prodCacheLoadPtr = MATRIX_POINTER_(prodCache, threadIdx);

    ////

    auto srcTexstep = imageTexstep;

    Point<Space> srcReadIdx = srcBase + threadIdx;
    Point<float32> srcReadTexPos = convertIndexToPos(srcReadIdx) * srcTexstep;

    PARALLEL_LOOP_2D_UNBASED
    (
        iX, iY, cacheSizeX, cacheSizeY, threadIdx.X, threadIdx.Y, threadCountX, threadCountY,

        {
            auto texPos = srcReadTexPos + point(float32(iX), float32(iY)) * srcTexstep;
            float32 mask = MASK_CHECK(tex2D(maskSampler, texPos), maskParams);
            float32 image = tex2D(imageSampler, texPos);

            *(maskCacheLoadPtr + iX + iY * maskCacheMemPitch) = mask;
            *(prodCacheLoadPtr + iX + iY * prodCacheMemPitch) = mask * image;
        }
    )

    ////

    devSyncThreads();

    //----------------------------------------------------------------
    //
    // Exit if not producing output.
    //
    //----------------------------------------------------------------

    if_not (vItemIsActive)
        return;

    //----------------------------------------------------------------
    //
    // Loop.
    //
    //----------------------------------------------------------------

    Point<Space> extendedIdx = threadIdx;
    extendedIdx.DIR(X, Y) *= downsampleFactor; // 2X bank conflicts, but not important.

    MatrixPtr(const float32) maskCachePtr = MATRIX_POINTER_(maskCache, extendedIdx);
    MatrixPtr(const float32) prodCachePtr = MATRIX_POINTER_(prodCache, extendedIdx);

    ////

    float32 filteredMask = 0;
    float32 filteredProd = 0;

    devUnrollLoop
    for (Space i = 0; i < filterSize; ++i)
    {
        float32 mask = maskCachePtr[i * DIR(1, maskCacheMemPitch)];
        float32 prod = prodCachePtr[i * DIR(1, prodCacheMemPitch)];

        float32 shape = PREP_PASTE(GABOR_BANK, DIR(ShapeX, ShapeY))[i];

        filteredMask += shape * mask;
        filteredProd += shape * prod;
    }

    ////

    storeNorm(dst, makeVec2(filteredMask, filteredProd));
}
#endif
GPUTOOL_2D_END_EX

//================================================================
//
// gaborDefaultFinalCached
//
//================================================================

GPUTOOL_2D_BEG_EX
(
    PREP_PASTE3(FUNCNAME, DefaultFinalCached, DIR(Hor, Ver)),
    GABOR_FINAL_CACHED_THREAD_COUNT,
    true,
    ((const DEFAULT_WEIGHTED_PIXEL, src, INTERP_NONE, INPUT_BORDER_MODE)),
    ((MASK_PIXEL, dstMask))
    ((DEFAULT_PIXEL, dstImage)),
    ((float32, dstMaskMinValue))
)
#if DEVCODE
{

    constexpr Space filterSize = PREP_PASTE(GABOR_BANK, DIR(SizeX, SizeY));
    COMPILE_ASSERT(!(COMPRESS_OCTAVES == 0) || (filterSize % 2 == 1));
    COMPILE_ASSERT(!(COMPRESS_OCTAVES >= 1) || (filterSize % 2 == 0));

    ////

    constexpr Space downsampleFactor = (1 << COMPRESS_OCTAVES);

    Point<Space> dstBase = vTileOrg;
    Point<Space> srcBase = dstBase;
    srcBase.DIR(X, Y) = mapDownsampleIndexToSource<downsampleFactor, filterSize>(srcBase.DIR(X, Y));

    //----------------------------------------------------------------
    //
    // ith thread first access: srcBase + i * downsampleFactor
    // ith thread last access: srcBase + i * downsampleFactor + filterSize - 1
    //
    // first access: srcBase
    // last access: srcBase + (threadCount-1) * downsampleFactor + filterSize - 1
    //
    // cache size: (threadCount-1) * downsampleFactor + filterSize
    //
    //----------------------------------------------------------------

    constexpr Space threadCountX = vTileSizeX;
    constexpr Space threadCountY = vTileSizeY;
    Point<Space> threadIdx = vTileMember;

    constexpr Space cacheSizeX = DIR((threadCountX-1) * downsampleFactor + filterSize, threadCountX);
    constexpr Space cacheSizeY = DIR(threadCountY, (threadCountY-1) * downsampleFactor + filterSize);

    //----------------------------------------------------------------
    //
    // Load src block.
    //
    //----------------------------------------------------------------

    devSramMatrixFor2dAccess(cache, float32_x2, cacheSizeX, cacheSizeY, threadCountX);

    ////

    MatrixPtr(float32_x2) cacheLoadPtr = MATRIX_POINTER_(cache, threadIdx);
    Point<Space> srcReadIdx = srcBase + threadIdx;
    Point<float32> srcReadTexPos = convertIndexToPos(srcReadIdx) * srcTexstep;

    ////

    PARALLEL_LOOP_2D_UNBASED
    (
        iX, iY, cacheSizeX, cacheSizeY, threadIdx.X, threadIdx.Y, threadCountX, threadCountY,
        *(cacheLoadPtr + iX + iY * cacheMemPitch) = tex2D(srcSampler, srcReadTexPos + point(float32(iX), float32(iY)) * srcTexstep);
    )

    ////

    devSyncThreads();

    //----------------------------------------------------------------
    //
    // Exit if not producing output value.
    //
    //----------------------------------------------------------------

    if_not (vItemIsActive)
        return;

    //----------------------------------------------------------------
    //
    // Filter.
    //
    //----------------------------------------------------------------

    Point<Space> extendedIdx = threadIdx;
    extendedIdx.DIR(X, Y) *= downsampleFactor; // 2X bank conflicts, but not important.
    MatrixPtr(const float32_x2) cacheStartPtr = MATRIX_POINTER_(cache, extendedIdx);

    ////

    float32_x2 sum = make_float32_x2(0, 0);

    devUnrollLoop
    for (Space i = 0; i < filterSize; ++i)
    {
        float32_x2 value = cacheStartPtr[i * DIR(1, cacheMemPitch)];
        float32 shape = PREP_PASTE(GABOR_BANK, DIR(ShapeX, ShapeY))[i];
        sum += shape * value;
    }

    //----------------------------------------------------------------
    //
    // Divide the filtered (image * mask) by the filtered mask.
    //
    //----------------------------------------------------------------

    auto result = nativeRecipZero(sum.x) * sum.y;
    storeNorm(dstImage, result);

    ////

    auto maskValue = sum.x;

    if_not (maskValue >= dstMaskMinValue)
        maskValue = 0;

    storeNorm(dstMask, maskValue);

}
#endif
GPUTOOL_2D_END_EX

//================================================================
//
// gaborDefaultFull
//
//================================================================

#if HOSTCODE

template <int=0>
stdbool PREP_PASTE3(FUNCNAME, DefaultFull, DIR(Hor, Ver))
(
    const GpuMatrix<const MASK_PIXEL>& srcMask, 
    const GpuMatrix<const INPUT_PIXEL>& srcImage, 
    const GpuMatrix<MASK_PIXEL>& dstMask,
    const GpuMatrix<DEFAULT_PIXEL>& dstImage,
    const MASK_PARAMS& maskParams,
    float32 dstMaskMinValue,
    stdPars(GpuProcessKit)
)
{
    REQUIRE(equalSize(srcMask, srcImage));
    auto srcSize = srcImage.size();

    REQUIRE(equalSize(dstMask, dstImage));
    auto dstSize = dstImage.size();

    ////

    Point<Space> tmpSize = srcSize;
    tmpSize.DIR(X, Y) = dstSize.DIR(X, Y);

    GPU_MATRIX_ALLOC(tmp, DEFAULT_WEIGHTED_PIXEL, tmpSize);

    ////

    auto initialFunc = PREP_PASTE3(FUNCNAME, DefaultInitialCached, DIR(Hor, Ver));

    require(initialFunc(srcMask, srcImage, tmp, maskParams, stdPass));

    ////

    auto finalFunc = PREP_PASTE3(FUNCNAME, DefaultFinalCached, DIR(Ver, Hor));

    require(finalFunc(tmp, dstMask, dstImage, dstMaskMinValue, stdPass));

    ////

    returnTrue;
}

#endif

//----------------------------------------------------------------

#endif // MASK_ENABLED
