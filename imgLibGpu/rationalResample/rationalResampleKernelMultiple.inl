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
// FUNCNAME
//
//================================================================

#if DEVCODE

template <typename Dst>
devDecl void PREP_PASTE4(FUNCNAME, DIR(Hor, Ver), RANK, Flex)(const ResampleParams<Dst>& o, Space taskIdx, devPars)
{
    using FloatType = VECTOR_REBASE(Dst, float32);
    
    ////

    const Space threadCountX = DIR(horThreadCountX, verThreadCountX);
    const Space threadCountY = DIR(horThreadCountY, verThreadCountY);

    Point<Space> packIdxBase = devGroupIdx * point(Space(threadCountX), Space(threadCountY));

    ////

    Point<Space> srcIdxBase = packIdxBase;
    srcIdxBase.DIR(X, Y) *= PACK_TO_SRC_FACTOR;
    srcIdxBase.DIR(X, Y) += FILTER_SRC_SHIFT;

    ////

    const Space filterSize = COMPILE_ARRAY_SIZE(FILTER0);

    #define TMP_MACRO(k, _) \
        COMPILE_ASSERT(COMPILE_ARRAY_SIZE(FILTER##k) == filterSize);

    PREP_FOR(PACK_SIZE, TMP_MACRO, _) 

    #undef TMP_MACRO

    //
    // srcIdx = FILTER_SRC_SHIFT + PACK_TO_SRC_FACTOR * packIdxBase + PACK_TO_SRC_FACTOR * threadIdx
    //
    // srcFirst: FILTER_SRC_SHIFT + PACK_TO_SRC_FACTOR * packIdxBase
    // srcLast = FILTER_SRC_SHIFT + PACK_TO_SRC_FACTOR * packIdxBase + PACK_TO_SRC_FACTOR * (threadCount-1) + (filterSize-1)
    //
    // count = (srcLast - srcFirst + 1) = PACK_TO_SRC_FACTOR * (threadCount-1) + filterSize
    //

    const Space cacheSizeX = DIR(PACK_TO_SRC_FACTOR * (threadCountX-1) + filterSize, threadCountX);
    const Space cacheSizeY = DIR(threadCountY, PACK_TO_SRC_FACTOR * (threadCountY-1) + filterSize);

    ////

    devSramMatrixFor2dAccess(cache, FloatType, cacheSizeX, cacheSizeY, threadCountX);

    //----------------------------------------------------------------
    //
    // Load src block!
    //
    //----------------------------------------------------------------

    auto cacheLoadPtr = cacheMemPtr + devThreadX + devThreadY * cacheMemPitch;

    ////

    Point<float32> srcPosRead = convertIndexToPos(srcIdxBase + devThreadIdx);

    #define LOAD_SRC_BLOCK(sampler) \
        \
        PARALLEL_LOOP_2D_UNBASED \
        ( \
            iX, iY, cacheSizeX, cacheSizeY, devThreadX, devThreadY, threadCountX, threadCountY, \
            *(cacheLoadPtr + iX + iY * cacheMemPitch) = tex2D(sampler, point(srcPosRead.X + iX, srcPosRead.Y + iY) * o.srcTexstep); \
        )

    ////

    #define TMP_MACRO(t, _) \
        if (taskIdx == t) LOAD_SRC_BLOCK(PREP_PASTE5(FUNCNAME, srcSampler, RANK, _task, t));

    PREP_FOR(TASK_COUNT, TMP_MACRO, _)

    #undef TMP_MACRO

    ////

    devSyncThreads();

    //----------------------------------------------------------------
    //
    // Convolve
    //
    //----------------------------------------------------------------

    #define TMP_MACRO(k, _) \
        FloatType result##k = convertNearest<FloatType>(0);

    PREP_FOR(PACK_SIZE, TMP_MACRO, _)

    #undef TMP_MACRO

    ////

    Point<Space> cacheReadPos = devThreadIdx;
    cacheReadPos.DIR(X, Y) *= PACK_TO_SRC_FACTOR;

    auto cacheReadPtr = cacheMemPtr + cacheReadPos.X + cacheReadPos.Y * cacheMemPitch;

    devUnrollLoop
    for_count (i, filterSize)
    {
        FloatType value = *cacheReadPtr;

        ////

        #define TMP_MACRO(k, _) \
            result##k += FILTER##k[i] * value;

        PREP_FOR(PACK_SIZE, TMP_MACRO, _)

        #undef TMP_MACRO

        ////

        DIR(cacheReadPtr++, cacheReadPtr += cacheMemPitch);
    }

    ////

    Point<Space> dstIdx = packIdxBase + devThreadIdx;
    dstIdx.DIR(X, Y) *= PACK_SIZE;

    //----------------------------------------------------------------
    //
    // 
    //
    //----------------------------------------------------------------

    MATRIX_EXPOSE_EX(o.dst[taskIdx], dst);

    ////

#ifdef OUTPUT_FACTOR

    #define TMP_MACRO(k, _) \
        result##k *= o.outputFactor;

    PREP_FOR(PACK_SIZE, TMP_MACRO, _)

    #undef TMP_MACRO

#endif

    //----------------------------------------------------------------
    //
    // 
    //
    //----------------------------------------------------------------

    #define TMP_MACRO(k, _) \
        { \
            Point<Space> dstIdx##k = dstIdx; dstIdx.DIR(X, Y)++; \
            bool dstValid##k = MATRIX_VALID_ACCESS_(dst, dstIdx##k); \
            auto dstPtr##k = MATRIX_POINTER_(dst, dstIdx##k); \
            if (dstValid##k) storeNorm(dstPtr##k, result##k); \
        }
    

    PREP_FOR(PACK_SIZE, TMP_MACRO, _);

    #undef TMP_MACRO

}

#endif

//================================================================
//
// FUNCNAME
//
//================================================================

#if DEVCODE

template <typename Dst>
devDecl void PREP_PASTE3(FUNCNAME, DIR(Hor, Ver), RANK)(const ResampleParams<Dst>& o, devPars)
{
    devDebugCheck(devGroupCountZ == TASK_COUNT);

    #define TMP_MACRO(t, _) \
        if (devGroupZ == t) {PREP_PASTE4(FUNCNAME, DIR(Hor, Ver), RANK, Flex)(o, t, devPass); return;}

    PREP_FOR(TASK_COUNT, TMP_MACRO, _)

    #undef TMP_MACRO
}

#endif
