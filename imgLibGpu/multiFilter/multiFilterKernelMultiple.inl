#undef DIR

//================================================================
//
// DIR
//
//================================================================

#if HORIZONTAL
    #define DIR(h, v) h
#else
    #define DIR(h, v) v
#endif

//================================================================
//
// FILTER
//
//================================================================

#undef FILTER

#define FILTER(k) \
    DIR(FILTER_HOR, FILTER_VER)(k)

//================================================================
//
// filterInterm
//
// Single input -> Multi output
//
//================================================================

#if DEVCODE

template <typename Dst>
devDecl void PREP_PASTE4(FUNCNAME, IntermFlex, DIR(Hor, Ver), RANK)(const IntermParams<Dst>& o, Space taskIdx, devPars)
{
    typedef VECTOR_REBASE(Dst, float32) FloatType;

    ////

    const Space threadCountX = DIR(horThreadCountX, verThreadCountX);
    const Space threadCountY = DIR(horThreadCountY, verThreadCountY);

    ////

    const Space filterSize = COMPILE_ARRAY_SIZE(FILTER(0));

    #define TMP_MACRO(k, _) \
        COMPILE_ASSERT(COMPILE_ARRAY_SIZE(FILTER(k)) == filterSize);

    PREP_FOR(FILTER_COUNT, TMP_MACRO, _) 

    #undef TMP_MACRO

    ////

    COMPILE_ASSERT(!(COMPRESS_OCTAVES == 0) || (filterSize % 2 == 1));
    COMPILE_ASSERT(!(COMPRESS_OCTAVES >= 1) || (filterSize % 2 == 0));

    ////

    Point<Space> dstBase = devGroupIdx * point(Space(threadCountX), Space(threadCountY));
    Point<Space> srcBase = dstBase;

    const Space downsampleFactor = (1 << COMPRESS_OCTAVES);
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

    const Space cacheSizeX = DIR((threadCountX-1) * downsampleFactor + filterSize, threadCountX);
    const Space cacheSizeY = DIR(threadCountY, (threadCountY-1) * downsampleFactor + filterSize);

    ////

    devSramMatrixFor2dAccess(cache, FloatType, cacheSizeX, cacheSizeY, threadCountX);

    //----------------------------------------------------------------
    //
    // Load src block
    //
    //----------------------------------------------------------------

    MatrixPtr(FloatType) cacheLoadPtr = MATRIX_POINTER_(cache, devThreadIdx);
    Point<Space> srcReadIdx = srcBase + devThreadIdx;
    Point<float32> srcReadTexPos = convertIndexToPos(srcReadIdx) * o.srcTexstep;

    ////

    #define LOAD_SRC_BLOCK_NOSYNC(sampler) \
        \
        PARALLEL_LOOP_2D_UNBASED \
        ( \
            iX, iY, cacheSizeX, cacheSizeY, devThreadIdx.X, devThreadIdx.Y, threadCountX, threadCountY, \
            *(cacheLoadPtr + iX + iY * cacheMemPitch) = tex2D(sampler, srcReadTexPos + point(float32(iX), float32(iY)) * o.srcTexstep); \
        )

    //
    // This switch should be compile-time.
    //

    #define TMP_MACRO(t, _) \
        if (t == taskIdx) {LOAD_SRC_BLOCK_NOSYNC(PREP_PASTE4(FUNCNAME, srcSampler, RANK, t));}

    PREP_FOR(TASK_COUNT, TMP_MACRO, _) 

    devSyncThreads();

    #undef TMP_MACRO
    #undef LOAD_SRC_BLOCK_NOSYNC

    //----------------------------------------------------------------
    //
    // Exit if not producing output
    //
    //----------------------------------------------------------------

    Point<Space> dstIdx = dstBase + devThreadIdx;

    if_not (matrixValidAccess(o.dstSize, dstIdx)) 
        return;

    //----------------------------------------------------------------
    //
    // Convolve
    //
    //----------------------------------------------------------------

    #define TMP_MACRO(k, _) \
        FloatType result##k = convertNearest<FloatType>(0);

    PREP_FOR(FILTER_COUNT, TMP_MACRO, _)

    #undef TMP_MACRO

    ////
  

    Point<Space> cacheReadPos = devThreadIdx;
    cacheReadPos.DIR(X, Y) *= downsampleFactor; // Potential bank conflicts, but not important
    MatrixPtr(const FloatType) cacheReadPtr = MATRIX_POINTER_(cache, cacheReadPos);

    ////

    devUnrollLoop
    for (Space i = 0; i < filterSize; ++i)
    {
        FloatType value = cacheReadPtr[i * DIR(1, cacheMemPitch)];

        ////

        #define TMP_MACRO(k, _) \
            result##k += FILTER(k)[i] * value;

        PREP_FOR(FILTER_COUNT, TMP_MACRO, _)

        #undef TMP_MACRO
    }

    //----------------------------------------------------------------
    //
    // Store
    //
    //----------------------------------------------------------------

    devDebugCheck(allv(o.dstSize >= 0));

    #define TMP_MACRO(k, _) \
        devDebugCheck(equalSize(o.dst[taskIdx][k], o.dstSize));

    PREP_FOR(FILTER_COUNT, TMP_MACRO, _)

    #undef TMP_MACRO

    ////

    #define TMP_MACRO(k, _) \
        \
        MATRIX_EXPOSE_EX(o.dst[taskIdx][k], dst##k); \
        MatrixPtr(Dst) dstPtr##k = MATRIX_POINTER_(dst##k, dstIdx); \
        storeNorm(dstPtr##k, result##k);

    PREP_FOR(FILTER_COUNT, TMP_MACRO, _)

    #undef TMP_MACRO

}

#endif

//================================================================
//
// intermFilterParallel
//
//================================================================

#if DEVCODE

template <typename Dst>
devDecl void PREP_PASTE4(FUNCNAME, IntermParallel, DIR(Hor, Ver), RANK)(const IntermParams<Dst>& o, devPars)
{
    #define TMP_MACRO(t, _) \
        if (t == devGroupZ) {PREP_PASTE4(FUNCNAME, IntermFlex, DIR(Hor, Ver), RANK)(o, t, devPass); return;} \

    PREP_FOR(TASK_COUNT, TMP_MACRO, _)

    #undef TMP_MACRO
}

#endif

//================================================================
//
// filterFinal
//
// Multi input -> Multi output
//
//================================================================

#if DEVCODE

template <typename Dst>
devDecl inline void PREP_PASTE4(FUNCNAME, FinalFlex, DIR(Hor, Ver), RANK)(const FinalParams<Dst>& o, Space taskIdx, devPars)
{
    typedef VECTOR_REBASE(Dst, float32) FloatType;

    ////

    const Space threadCountX = DIR(horThreadCountX, verThreadCountX);
    const Space threadCountY = DIR(horThreadCountY, verThreadCountY);

    ////

    const Space filterSize = COMPILE_ARRAY_SIZE(FILTER(0));

    #define TMP_MACRO(k, _) \
        COMPILE_ASSERT(COMPILE_ARRAY_SIZE(FILTER(k)) == filterSize);

    PREP_FOR(FILTER_COUNT, TMP_MACRO, _) 

    #undef TMP_MACRO

    ////

    COMPILE_ASSERT(!(COMPRESS_OCTAVES == 0) || (filterSize % 2 == 1));
    COMPILE_ASSERT(!(COMPRESS_OCTAVES >= 1) || (filterSize % 2 == 0));

    ////

    Point<Space> dstBase = devGroupIdx * point(Space(threadCountX), Space(threadCountY));
    Point<Space> srcBase = dstBase;

    const Space downsampleFactor = (1 << COMPRESS_OCTAVES);
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

    const Space cacheSizeX = DIR((threadCountX-1) * downsampleFactor + filterSize, threadCountX);
    const Space cacheSizeY = DIR(threadCountY, (threadCountY-1) * downsampleFactor + filterSize);

    ////

    devSramMatrixFor2dAccess(cache, FloatType, cacheSizeX, cacheSizeY, threadCountX);

    //----------------------------------------------------------------
    //
    // Load src block
    //
    //----------------------------------------------------------------

    MatrixPtr(FloatType) cacheLoadPtr = MATRIX_POINTER_(cache, devThreadIdx);
    Point<Space> srcLoadIdx = srcBase + devThreadIdx;
    Point<float32> srcLoadTexPos = convertIndexToPos(srcLoadIdx) * o.srcTexstep;

    ////

    #define LOAD_SRC_BLOCK_FLEX_NOSYNC(t, k) \
        \
        PARALLEL_LOOP_2D_UNBASED \
        ( \
            iX, iY, cacheSizeX, cacheSizeY, devThreadIdx.X, devThreadIdx.Y, threadCountX, threadCountY, \
            *(cacheLoadPtr + iX + iY * cacheMemPitch) = tex2D(PREP_PASTE5(FUNCNAME, intermSampler, k, RANK, t), srcLoadTexPos + point(float32(iX), float32(iY)) * o.srcTexstep); \
        ) \

    ////

    #define LOAD_SRC_BLOCK_ITER(t, k) \
        if (taskIdx == t) LOAD_SRC_BLOCK_FLEX_NOSYNC(t, k);

    #define LOAD_SRC_BLOCK_NOSYNC(k) \
        PREP_FOR(TASK_COUNT, LOAD_SRC_BLOCK_ITER, k) /* should reduce to one branch at compile-time */

    //----------------------------------------------------------------
    //
    // Convolve
    //
    //----------------------------------------------------------------
  

    Point<Space> cacheReadPos = devThreadIdx;
    cacheReadPos.DIR(X, Y) *= downsampleFactor; // Potential bank conflicts, but not important
    MatrixPtr(const FloatType) cacheReadPtr = MATRIX_POINTER_(cache, cacheReadPos);

    ////

    #define TMP_MACRO(k, _) \
        \
        FloatType result##k = convertNearest<FloatType>(0); \
        \
        /* Finish using cache */ \
        if (k != 0) devSyncThreads(); \
        \
        /* Load block */ \
        LOAD_SRC_BLOCK_NOSYNC(k); \
        devSyncThreads(); \
        \
        devUnrollLoop \
        for (Space i = 0; i < filterSize; ++i) \
        { \
            FloatType value = cacheReadPtr[i * DIR(1, cacheMemPitch)]; \
            result##k += FILTER(k)[i] * value; \
        }

    PREP_FOR1(FILTER_COUNT, TMP_MACRO, _)

    #undef TMP_MACRO
    

    #undef LOAD_SRC_BLOCK_NOSYNC
    #undef LOAD_SRC_BLOCK_ITER
    #undef LOAD_SRC_BLOCK_FLEX_NOSYNC

    //----------------------------------------------------------------
    //
    // Exit if not producing output
    //
    //----------------------------------------------------------------

    Point<Space> dstIdx = dstBase + devThreadIdx;

    if_not (matrixValidAccess(o.dstSize, dstIdx)) 
        return;

    //----------------------------------------------------------------
    //
    // Store
    //
    //----------------------------------------------------------------

#ifndef LINEAR_COMBINATION

    #define TMP_MACRO(k, _) \
        MATRIX_EXPOSE_EX(o.dst[taskIdx][k], dst##k); \
        MatrixPtr(Dst) dstPtr##k = MATRIX_POINTER_(dst##k, dstIdx); \
        storeNorm(dstPtr##k, result##k);

    PREP_FOR(FILTER_COUNT, TMP_MACRO, _)

    #undef TMP_MACRO

#else

    FloatType result = convertNearest<FloatType>(0);

    ////

    #define TMP_MACRO(k, _) \
        result += o.filterMixCoeffs[k] * result##k;

    PREP_FOR(FILTER_COUNT, TMP_MACRO, _)

    #undef TMP_MACRO

    ////

    float32 dstMixCoeff = o.dstMixCoeff[taskIdx];

    if (dstMixCoeff)
    {
        MATRIX_EXPOSE_EX(o.dstMixImage[taskIdx], dstMixImage);
        result += dstMixCoeff * loadNorm(MATRIX_POINTER_(dstMixImage, dstIdx));
    }

    ////

    devDebugCheck(allv(o.dstSize >= 0));
    devDebugCheck(equalSize(o.dst[taskIdx], o.dstSize));

    MATRIX_EXPOSE_EX(o.dst[taskIdx], dst);
    storeNorm(MATRIX_POINTER_(dst, dstIdx), result);

#endif

}

#endif

//================================================================
//
// finalFilterParallel
//
//================================================================

#if DEVCODE

template <typename Dst>
devDecl inline void PREP_PASTE4(FUNCNAME, FinalParallel, DIR(Hor, Ver), RANK)(const FinalParams<Dst>& o, devPars)
{
    #define TMP_MACRO(t, _) \
        if (t == devGroupZ) {PREP_PASTE4(FUNCNAME, FinalFlex, DIR(Hor, Ver), RANK)(o, t, devPass); return;}
        

    PREP_FOR(TASK_COUNT, TMP_MACRO, _)

    #undef TMP_MACRO
}

#endif

