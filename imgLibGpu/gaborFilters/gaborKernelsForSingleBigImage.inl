//================================================================
//
// Input parameters.
//
//================================================================

#if !(defined(INPUT_PIXEL) && defined(COMPLEX_PIXEL))
    #error Type parameters are required.
#endif

//----------------------------------------------------------------

#if !defined(FUNCNAME)
    #error
#endif

//----------------------------------------------------------------

#if !(defined(GABOR_BANK) && defined(ORIENT_COUNT) && defined(COMPRESS_OCTAVES) && defined(GABOR_BORDER_MODE))
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

//================================================================
//
// gaborBankFirstSimple
//
//================================================================

GPUTOOL_2D_BEG
(
    PREP_PASTE3(FUNCNAME, FirstSimple, DIR(Hor, Ver)),
    ((const INPUT_PIXEL, src, INTERP_NEAREST, GABOR_BORDER_MODE))
    ((const float32_x2, circleTable, INTERP_LINEAR, BORDER_WRAP)),
    GPUTOOL_INDEXED_NAME(ORIENT_COUNT, COMPLEX_PIXEL, dst),
    PREP_EMPTY
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
        sum##k = complexMul(sum##k, devTex2D(circleTableSampler, -PREP_PASTE3(GABOR_BANK, Freq, k).DIR(x, y) * filterCenter, 0)); \
        storeNorm(dst##k, sum##k);

    PREP_FOR(ORIENT_COUNT, TMP_MACRO, _)

    #undef TMP_MACRO

}
#endif
GPUTOOL_2D_END

//================================================================
//
// gaborBankFirstCached
//
//================================================================

GPUTOOL_2D_BEG_EX
(
    PREP_PASTE3(FUNCNAME, FirstCached, DIR(Hor, Ver)),
    DIR((64, 4), (32, 8)),
    true,
    ((const INPUT_PIXEL, src, INTERP_NEAREST, GABOR_BORDER_MODE))
    ((const float32_x2, circleTable, INTERP_LINEAR, BORDER_WRAP)),
    GPUTOOL_INDEXED_NAME(ORIENT_COUNT, COMPLEX_PIXEL, dst),
    PREP_EMPTY
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
    extendedIdx.DIR(X, Y) *= downsampleFactor; // 2X bank conflicts, but not important
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
        sum##k = complexMul(sum##k, devTex2D(circleTableSampler, -PREP_PASTE3(GABOR_BANK, Freq, k).DIR(x, y) * filterCenter, 0)); \
        storeNorm(dst##k, sum##k);

    PREP_FOR(ORIENT_COUNT, TMP_MACRO, _)

    #undef TMP_MACRO

}
#endif
GPUTOOL_2D_END_EX

//================================================================
//
// GABOR_FINAL_FACTOR
//
//================================================================

#define GABOR_FINAL_FACTOR 4.f

//================================================================
//
// gaborBankLastSimple
//
//================================================================

GPUTOOL_2D_BEG
(
    PREP_PASTE3(FUNCNAME, LastSimple, DIR(Hor, Ver)),
    GPUTOOL_INDEXED_SAMPLER(ORIENT_COUNT, const COMPLEX_PIXEL, src, INTERP_NONE, GABOR_BORDER_MODE)
    ((const float32_x2, circleTable, INTERP_LINEAR, BORDER_WRAP)),
    GPUTOOL_INDEXED_NAME(ORIENT_COUNT, COMPLEX_PIXEL, dst),
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
        sum##k = complexMul(sum##k, devTex2D(circleTableSampler, -PREP_PASTE3(GABOR_BANK, Freq, k).DIR(x, y) * filterCenter, 0)); \
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
// gaborBankLastCached
//
//================================================================

GPUTOOL_2D_BEG_EX
(
    PREP_PASTE3(FUNCNAME, LastCached, DIR(Hor, Ver)),
    DIR((32, 4), (32, 8)),
    true,
    GPUTOOL_INDEXED_SAMPLER(ORIENT_COUNT, const COMPLEX_PIXEL, src, INTERP_NONE, GABOR_BORDER_MODE)
    ((const float32_x2, circleTable, INTERP_LINEAR, BORDER_WRAP)),
    GPUTOOL_INDEXED_NAME(ORIENT_COUNT, COMPLEX_PIXEL, dst),
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

    //
    // For 2X downsampling, the access gives 2X bank conflicts, 
    // but it doesn't influence performance.
    //

    Point<Space> extendedIdx = threadIdx;
    extendedIdx.DIR(X, Y) *= downsampleFactor; 
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
        sum##k = complexMul(sum##k, devTex2D(circleTableSampler, -PREP_PASTE3(GABOR_BANK, Freq, k).DIR(x, y) * filterCenter, 0)); \
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
// gaborBankFlex
//
//================================================================

#if HOSTCODE

template <int=0>
stdbool PREP_PASTE3(FUNCNAME, Flex, DIR(Hor, Ver))
(
    const GpuMatrix<const INPUT_PIXEL>& src, 
    const GpuMatrix<const float32_x2>& circleTable,
    const GpuLayeredMatrix<COMPLEX_PIXEL>& dst,
    const POSTPROCESS_PARAMS& postprocessParams,
    bool simpleVersion,
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
            simpleVersion ?
                PREP_PASTE3(FUNCNAME, FirstSimple, DIR(Hor, Ver)) :
                PREP_PASTE3(FUNCNAME, FirstCached, DIR(Hor, Ver))
        )
        (
            src, 
            circleTable, 
            GPU_LAYERED_MATRIX_PASS(ORIENT_COUNT, tmp), 
            stdPass
        )
    );

    ////

    require
    (
        ( 
            simpleVersion ? 
            PREP_PASTE3(FUNCNAME, LastSimple, DIR(Ver, Hor)) :
            PREP_PASTE3(FUNCNAME, LastCached, DIR(Ver, Hor))
        )
        (
            GPU_LAYERED_MATRIX_PASS(ORIENT_COUNT, tmp), 
            circleTable, 
            GPU_LAYERED_MATRIX_PASS(ORIENT_COUNT, dst), 
            postprocessParams,
            stdPass
        )
    );

    ////

    returnTrue;
}

#endif
