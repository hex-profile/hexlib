//================================================================
//
// Input parameters.
//
//================================================================

#undef DIR

#if !defined(HORIZONTAL)
    #error HORIZONTAL should be defined to 0 or 1
#elif HORIZONTAL
    #define DIR(h, v) h
#else
    #define DIR(h, v) v
#endif

//----------------------------------------------------------------

#if !(defined(GABOR_ENABLED) && defined(ENVELOPE_ENABLED))
    #error
#endif

//----------------------------------------------------------------

#if !(defined(FUNCNAME) && defined(COMPRESS_OCTAVES) && defined(GABOR_BANK))
    #error
#endif

//----------------------------------------------------------------

#if GABOR_ENABLED

    #if !(defined(GABOR_INPUT_PIXEL) && defined(GABOR_COMPLEX_PIXEL))
        #error Type parameters are required.
    #endif

    #if !(defined(GABOR_ORIENT_COUNT) && defined(GABOR_BORDER_MODE))
        #error
    #endif

    #if !(defined(GABOR_PARAMS))
        #error
    #endif

    #if !(defined(GABOR_PREPROCESS_IMAGES) && defined(GABOR_PREPROCESS))
        #error
    #endif

    #if !(defined(GABOR_POSTPROCESS_IMAGES) && defined(GABOR_POSTPROCESS))
        #error
    #endif

#endif

//----------------------------------------------------------------

#if ENVELOPE_ENABLED

    #if !(defined(ENVELOPE_BORDER_MODE) && defined(ENVELOPE_PARAMS))
        #error
    #endif

    #if !(defined(ENVELOPE_INPUT_IMAGES) && defined(ENVELOPE_LOAD))
        #error
    #endif

    #if !(defined(ENVELOPE_OUTPUT_IMAGES) && defined(ENVELOPE_STORE))
        #error
    #endif

    #if !defined(ENVELOPE_INTERM_PIXEL)
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

#if GABOR_ENABLED

//================================================================
//
// GABOR_DECLARE_SAMPLER
// GABOR_DECLARE_MATRIX
//
//================================================================

#define GABOR_DECLARE_SAMPLER(Type, name, _) \
    ((Type, name, INTERP_NONE, GABOR_BORDER_MODE))

#define GABOR_DECLARE_MATRIX(Type, name, _) \
    ((Type, name))

#define GABOR_DECLARE_MATRIX_PARAM(Type, name, _) \
    const GpuMatrix<Type>& name,

#define GABOR_PASS_MATRIX_PARAM(Type, name, _) \
    name,

//================================================================
//
// gaborProcessInitialSimple
//
//================================================================

GPUTOOL_2D_BEG
(
    PREP_PASTE3(FUNCNAME, ProcessInitialSimple, DIR(Hor, Ver)),

    ((const GABOR_INPUT_PIXEL, src, INTERP_NONE, GABOR_BORDER_MODE))
    PREP_LIST_FOREACH_PAIR(GABOR_PREPROCESS_IMAGES (o), GABOR_DECLARE_SAMPLER, _)
    ((const float32_x2, circleTable, INTERP_LINEAR, BORDER_WRAP)),

    GPUTOOL_INDEXED_NAME(GABOR_ORIENT_COUNT, GABOR_COMPLEX_PIXEL, dst),

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
        auto sum##k = zeroOf<float32_x2>();

    PREP_FOR(GABOR_ORIENT_COUNT, TMP_MACRO, _)

    #undef TMP_MACRO

    ////

    Point<float32> srcReadTexPos = convertIndexToPos(srcIdx) * srcTexstep;

    ////

    devUnrollLoop
    for (Space i = 0; i < filterSize; ++i)
    {
        Point<Space> ofs = point(0);
        ofs.DIR(X, Y) = i;

        auto texPos = srcReadTexPos + convertFloat32(ofs) * srcTexstep;
        auto value = tex2D(srcSampler, texPos);
        GABOR_PREPROCESS(value, texPos)

        #define TMP_MACRO(k, _) \
            sum##k += PREP_PASTE3(GABOR_BANK, DIR(DataX, DataY), k)[i] * value;

        PREP_FOR(GABOR_ORIENT_COUNT, TMP_MACRO, PREP_EMPTY)

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

    PREP_FOR(GABOR_ORIENT_COUNT, TMP_MACRO, _)

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

    ((const GABOR_INPUT_PIXEL, src, INTERP_NONE, GABOR_BORDER_MODE))
    PREP_LIST_FOREACH_PAIR(GABOR_PREPROCESS_IMAGES (o), GABOR_DECLARE_SAMPLER, _)
    ((const float32_x2, circleTable, INTERP_LINEAR, BORDER_WRAP)),

    GPUTOOL_INDEXED_NAME(GABOR_ORIENT_COUNT, GABOR_COMPLEX_PIXEL, dst),

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

        {
            auto texPos = srcReadTexPos + point(float32(iX), float32(iY)) * srcTexstep;
            auto value = tex2D(srcSampler, texPos);
            GABOR_PREPROCESS(value, texPos);
            *(cacheLoadPtr + iX + iY * cacheMemPitch) = value;
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

    #define TMP_MACRO(k, _) \
        auto sum##k = zeroOf<float32_x2>();

    PREP_FOR(GABOR_ORIENT_COUNT, TMP_MACRO, _)

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

        PREP_FOR(GABOR_ORIENT_COUNT, TMP_MACRO, _)

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

    PREP_FOR(GABOR_ORIENT_COUNT, TMP_MACRO, _)

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

    GPUTOOL_INDEXED_SAMPLER(GABOR_ORIENT_COUNT, const GABOR_COMPLEX_PIXEL, src, INTERP_NONE, GABOR_BORDER_MODE)
    ((const float32_x2, circleTable, INTERP_LINEAR, BORDER_WRAP)),

    PREP_LIST_FOREACH_PAIR(GABOR_POSTPROCESS_IMAGES (o), GABOR_DECLARE_MATRIX, _)
    GPUTOOL_INDEXED_NAME(GABOR_ORIENT_COUNT, GABOR_COMPLEX_PIXEL, dst),

    ((bool, demodulateOutput))
    ((GABOR_PARAMS, params))
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
        auto sum##k = zeroOf<float32_x2>(); \
        \
        devUnrollLoop \
        for (Space i = 0; i < filterSize; ++i) \
        { \
            Point<Space> ofs = point(0); \
            ofs.DIR(X, Y) = i; \
            \
            auto value = tex2D(src##k##Sampler, srcReadTexPos + convertFloat32(ofs) * src0Texstep); \
            sum##k = complexMad(sum##k, PREP_PASTE3(GABOR_BANK, DIR(DataX, DataY), k)[i], value); \
        }

    PREP_FOR(GABOR_ORIENT_COUNT, TMP_MACRO, _)

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

    PREP_FOR(GABOR_ORIENT_COUNT, TMP_MACRO, _)

    #undef TMP_MACRO

    //----------------------------------------------------------------
    //
    // Postprocess.
    //
    //----------------------------------------------------------------

    GABOR_POSTPROCESS(sum)

    //----------------------------------------------------------------
    //
    // Store.
    //
    //----------------------------------------------------------------

    #define TMP_MACRO(k, _) \
        storeNorm(dst##k, sum##k);

    PREP_FOR(GABOR_ORIENT_COUNT, TMP_MACRO, _)

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

    GPUTOOL_INDEXED_SAMPLER(GABOR_ORIENT_COUNT, const GABOR_COMPLEX_PIXEL, src, INTERP_NONE, GABOR_BORDER_MODE)
    ((const float32_x2, circleTable, INTERP_LINEAR, BORDER_WRAP)),

    PREP_LIST_FOREACH_PAIR(GABOR_POSTPROCESS_IMAGES (o), GABOR_DECLARE_MATRIX, _)
    GPUTOOL_INDEXED_NAME(GABOR_ORIENT_COUNT, GABOR_COMPLEX_PIXEL, dst),

    ((bool, demodulateOutput))
    ((GABOR_PARAMS, params))
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
        auto sum##k = zeroOf<float32_x2>(); \
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
            auto value = cacheStartPtr[i * DIR(1, cacheMemPitch)]; \
            sum##k = complexMad(sum##k, PREP_PASTE3(GABOR_BANK, DIR(DataX, DataY), k)[i], value); \
        }

    PREP_FOR(GABOR_ORIENT_COUNT, TMP_MACRO, _)

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

    PREP_FOR(GABOR_ORIENT_COUNT, TMP_MACRO, _)

    #undef TMP_MACRO

    //----------------------------------------------------------------
    //
    // Postprocess.
    //
    //----------------------------------------------------------------

    GABOR_POSTPROCESS(sum)

    //----------------------------------------------------------------
    //
    // Store.
    //
    //----------------------------------------------------------------

    #define TMP_MACRO(k, _) \
        storeNorm(dst##k, sum##k);

    PREP_FOR(GABOR_ORIENT_COUNT, TMP_MACRO, _)

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
    const GpuMatrix<const GABOR_INPUT_PIXEL>& src, 
    PREP_LIST_FOREACH_PAIR(GABOR_PREPROCESS_IMAGES (o), GABOR_DECLARE_MATRIX_PARAM, _)
    const GpuMatrix<const float32_x2>& circleTable,
    PREP_LIST_FOREACH_PAIR(GABOR_POSTPROCESS_IMAGES (o), GABOR_DECLARE_MATRIX_PARAM, _)
    const GpuLayeredMatrix<GABOR_COMPLEX_PIXEL>& dst,
    bool demodulateOutput,
    const GABOR_PARAMS& params,
    bool uncachedVersion,
    stdPars(GpuProcessKit)
)
{
    //----------------------------------------------------------------
    //
    // Check image sizes.
    //
    //----------------------------------------------------------------

    auto srcSize = src.size();

    ////

    #define TMP_MACRO(Type, name, _) \
        REQUIRE(equalSize(srcSize, name.size()));

    PREP_LIST_FOREACH_PAIR(GABOR_PREPROCESS_IMAGES (o), TMP_MACRO, _)

    #undef TMP_MACRO

    //----------------------------------------------------------------
    //
    // Destination size.
    //
    //----------------------------------------------------------------

    auto dstSize = dst.size();

    ////

    #define TMP_MACRO(Type, name, _) \
        REQUIRE(equalSize(dstSize, name.size()));

    PREP_LIST_FOREACH_PAIR(GABOR_POSTPROCESS_IMAGES (o), TMP_MACRO, _)

    #undef TMP_MACRO

    //----------------------------------------------------------------
    //
    // Initial pass.
    //
    //----------------------------------------------------------------

    Point<Space> tmpSize = srcSize;
    tmpSize.DIR(X, Y) = dstSize.DIR(X, Y);

    ////

    GPU_LAYERED_MATRIX_ALLOC(tmp, GABOR_COMPLEX_PIXEL, GABOR_ORIENT_COUNT, tmpSize);

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
            PREP_LIST_FOREACH_PAIR(GABOR_PREPROCESS_IMAGES (o), GABOR_PASS_MATRIX_PARAM, _)
            circleTable, 
            GPU_LAYERED_MATRIX_PASS(GABOR_ORIENT_COUNT, tmp), 
            demodulateOutput,
            stdPass
        )
    );

    //----------------------------------------------------------------
    //
    // Final pass.
    //
    //----------------------------------------------------------------

    require
    (
        ( 
            uncachedVersion ? 
            PREP_PASTE3(FUNCNAME, ProcessFinalSimple, DIR(Ver, Hor)) :
            PREP_PASTE3(FUNCNAME, ProcessFinalCached, DIR(Ver, Hor))
        )
        (
            GPU_LAYERED_MATRIX_PASS(GABOR_ORIENT_COUNT, tmp), 
            circleTable, 
            PREP_LIST_FOREACH_PAIR(GABOR_POSTPROCESS_IMAGES (o), GABOR_PASS_MATRIX_PARAM, _)
            GPU_LAYERED_MATRIX_PASS(GABOR_ORIENT_COUNT, dst), 
            demodulateOutput,
            params,
            stdPass
        )
    );

    ////

    returnTrue;
}

#endif

//----------------------------------------------------------------

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
// 2D weighted average of defined pixels within the weight window of
// Gabor envelope (Gaussian ball).
//
// * When computing each Gabor filter, replace undefined input pixels with 
// the default value. The replacement cannot be done inside the input image, 
// because the same input pixel may have different default values 
// when it is used for different Gabor positions.
//
// The above description prevents separable filtering if implemented directly.
// To keep filtering separable, the following method is used:
//
// Assume we don't know the default input value at the filtering stage, 
// let's denote it by variable D. 
//
// For unconditional input, the filtered value is sum(Fi * Vi)
// where F is the filter and V is value.
//
// For conditional input, handled by replacing invalid pixels with value D,
// the input value is (Mi * Vi + (1 - Mi) * D), where M is mask (or probability).
//
// sum(Fi * (Mi * Vi + (1 - Mi) * D))
// ==
// sum(Fi * Mi * Vi) + D * sum(Fi * 1) - sum(Fi * Mi)
//
// So, the result of any linear filter, including complex Gabor filter, splits into three sums: 
// (Filtered product of image and mask) + (Filtered 1) - D * (Filtered mask).
//
// For Gabor filters, filtered 1 is zero, so the result is:
// (Gabor-filtered product of image and mask) - D * (Gabor-filtered mask).
//
// Both filters can be computed separably, as well as the image of default value D, which is 
// (envelope-filtered product of image and mask) / (envelope-filtered mask).
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

#if ENVELOPE_ENABLED

//================================================================
//
// ENVELOPE_DECLARE_SAMPLER
// ENVELOPE_DECLARE_MATRIX
//
//================================================================

#define ENVELOPE_DECLARE_SAMPLER(Type, name, _) \
    ((Type, name, INTERP_NONE, ENVELOPE_BORDER_MODE))

#define ENVELOPE_DECLARE_MATRIX(Type, name, _) \
    ((Type, name))

//================================================================
//
// gaborEnvelopeInitialCached
//
//================================================================

GPUTOOL_2D_BEG_EX
(
    PREP_PASTE3(FUNCNAME, EnvelopeInitialCached, DIR(Hor, Ver)),
    GABOR_INITIAL_CACHED_THREAD_COUNT,
    true,
    
    PREP_LIST_FOREACH_PAIR(ENVELOPE_INPUT_IMAGES (o), ENVELOPE_DECLARE_SAMPLER, _),
    ((ENVELOPE_INTERM_PIXEL, dst)),
    ((ENVELOPE_PARAMS, params))
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

    using ValueType = float32;

    ////

    devSramMatrixFor2dAccess(valueCache, ValueType, cacheSizeX, cacheSizeY, threadCountX);
    auto valueCacheLoadPtr = MATRIX_POINTER_(valueCache, threadIdx);

    ////

    Point<float32> srcTexstep{};

    #define TMP_MACRO(Type, name, _) \
        srcTexstep = name##Texstep;

    PREP_LIST_FOREACH_PAIR(ENVELOPE_INPUT_IMAGES (o), TMP_MACRO, _)

    #undef TMP_MACRO

    ////

    Point<Space> srcReadIdx = srcBase + threadIdx;
    Point<float32> srcReadTexPos = convertIndexToPos(srcReadIdx) * srcTexstep;

    PARALLEL_LOOP_2D_UNBASED
    (
        iX, iY, cacheSizeX, cacheSizeY, threadIdx.X, threadIdx.Y, threadCountX, threadCountY,

        {
            auto texPos = srcReadTexPos + point(float32(iX), float32(iY)) * srcTexstep;
            *(valueCacheLoadPtr + iX + iY * valueCacheMemPitch) = ENVELOPE_LOAD(texPos, params);
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

    MatrixPtr(const ValueType) valueCachePtr = MATRIX_POINTER_(valueCache, extendedIdx);

    ////

    ValueType filteredValue = convertNearest<ValueType>(0);

    devUnrollLoop
    for (Space i = 0; i < filterSize; ++i)
    {
        auto value = valueCachePtr[i * DIR(1, valueCacheMemPitch)];
        auto shape = PREP_PASTE(GABOR_BANK, DIR(ShapeX, ShapeY))[i];
        filteredValue += shape * value;
    }

    ////

    storeNorm(dst, filteredValue);
}
#endif
GPUTOOL_2D_END_EX

//================================================================
//
// gaborEnvelopeFinalCached
//
//================================================================

GPUTOOL_2D_BEG_EX
(
    PREP_PASTE3(FUNCNAME, EnvelopeFinalCached, DIR(Hor, Ver)),
    GABOR_FINAL_CACHED_THREAD_COUNT,
    true,
    
    ((const ENVELOPE_INTERM_PIXEL, src, INTERP_NONE, ENVELOPE_BORDER_MODE)),
    PREP_LIST_FOREACH_PAIR(ENVELOPE_OUTPUT_IMAGES (o), ENVELOPE_DECLARE_MATRIX, _),
    ((ENVELOPE_PARAMS, params))
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

    using ValueType = float32;

    devSramMatrixFor2dAccess(cache, ValueType, cacheSizeX, cacheSizeY, threadCountX);

    ////

    MatrixPtr(ValueType) cacheLoadPtr = MATRIX_POINTER_(cache, threadIdx);
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
    MatrixPtr(const ValueType) cacheStartPtr = MATRIX_POINTER_(cache, extendedIdx);

    ////

    ValueType filteredValue = convertNearest<ValueType>(0);

    devUnrollLoop
    for (Space i = 0; i < filterSize; ++i)
    {
        ValueType value = cacheStartPtr[i * DIR(1, cacheMemPitch)];
        float32 shape = PREP_PASTE(GABOR_BANK, DIR(ShapeX, ShapeY))[i];
        filteredValue += shape * value;
    }

    //----------------------------------------------------------------
    //
    // Divide the filtered (image * mask) by the filtered mask.
    //
    //----------------------------------------------------------------

    ENVELOPE_STORE(filteredValue);

}
#endif
GPUTOOL_2D_END_EX

//================================================================
//
// gaborEnvelopeFull
//
//================================================================

#if HOSTCODE

template <int=0>
GPUTOOL_2D_PROTO
(
    PREP_PASTE3(FUNCNAME, EnvelopeFull, DIR(Hor, Ver)),
    PREP_LIST_FOREACH_PAIR(ENVELOPE_INPUT_IMAGES (o), ENVELOPE_DECLARE_SAMPLER, _),
    PREP_LIST_FOREACH_PAIR(ENVELOPE_OUTPUT_IMAGES (o), ENVELOPE_DECLARE_MATRIX, _),
    ((ENVELOPE_PARAMS, params))
)
{

    //----------------------------------------------------------------
    //
    // Source size.
    //
    //----------------------------------------------------------------

    Point<Space> srcSize{};

    #define TMP_MACRO(Type, name, _) \
        srcSize = name##Matrix.size();

    PREP_LIST_FOREACH_PAIR(ENVELOPE_INPUT_IMAGES (o), TMP_MACRO, _)

    #undef TMP_MACRO

    ////

    #define TMP_MACRO(Type, name, _) \
        REQUIRE(equalSize(srcSize, name##Matrix.size()));

    PREP_LIST_FOREACH_PAIR(ENVELOPE_INPUT_IMAGES (o), TMP_MACRO, _)

    #undef TMP_MACRO

    //----------------------------------------------------------------
    //
    // Destination size.
    //
    //----------------------------------------------------------------

    Point<Space> dstSize{};

    #define TMP_MACRO(Type, name, _) \
        dstSize = name##Matrix.size();

    PREP_LIST_FOREACH_PAIR(ENVELOPE_OUTPUT_IMAGES (o), TMP_MACRO, _)

    #undef TMP_MACRO

    ////

    #define TMP_MACRO(Type, name, _) \
        REQUIRE(equalSize(dstSize, name##Matrix.size()));

    PREP_LIST_FOREACH_PAIR(ENVELOPE_OUTPUT_IMAGES (o), TMP_MACRO, _)

    #undef TMP_MACRO

    //----------------------------------------------------------------
    //
    // Temporary buffer.
    //
    //----------------------------------------------------------------

    Point<Space> tmpSize = srcSize;
    tmpSize.DIR(X, Y) = dstSize.DIR(X, Y);

    GPU_MATRIX_ALLOC(tmp, ENVELOPE_INTERM_PIXEL, tmpSize);

    //----------------------------------------------------------------
    //
    // Initial pass.
    //
    //----------------------------------------------------------------

    auto initialFunc = PREP_PASTE3(FUNCNAME, EnvelopeInitialCached, DIR(Hor, Ver));

    #define TMP_MACRO(Type, name, _) \
        name##Matrix

    require(initialFunc(PREP_LIST_ENUM_PAIR(ENVELOPE_INPUT_IMAGES (o), TMP_MACRO, _), tmp, params, stdPass));

    #undef TMP_MACRO

    //----------------------------------------------------------------
    //
    // Final past.
    //
    //----------------------------------------------------------------

    auto finalFunc = PREP_PASTE3(FUNCNAME, EnvelopeFinalCached, DIR(Ver, Hor));

    #define TMP_MACRO(Type, name, _) \
        name##Matrix

    require(finalFunc(tmp, PREP_LIST_ENUM_PAIR(ENVELOPE_OUTPUT_IMAGES (o), TMP_MACRO, _), params, stdPass));

    #undef TMP_MACRO

    ////

    returnTrue;
}

#endif

//----------------------------------------------------------------

#endif // ENVELOPE_ENABLED
