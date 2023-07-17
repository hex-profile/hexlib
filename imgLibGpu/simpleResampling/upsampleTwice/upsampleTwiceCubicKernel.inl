//================================================================
//
// upsampleTwiceKernel
//
//================================================================

#if DEVCODE

devDefineKernel(PREP_PASTE(upsampleTwiceKernel, DST), UpsampleParams<DST>, o)
{
    typedef VECTOR_REBASE_(DST, float32) FloatType;

    MATRIX_EXPOSE_EX(o.dst, dst);

    Space srcBaseX = devGroupX * threadCountX - 1;
    Space srcBaseY = devGroupY * threadCountY - 1;

    //----------------------------------------------------------------
    //
    // SRAM matrix for source tile: 1 + size + 2
    //
    //----------------------------------------------------------------

    const Space extraL = 1;
    const Space extraR = 2;
    const Space extraLR = extraL + extraR;

    devSramMatrixFor2dAccess(srcBuffer, FloatType, threadCountX + extraLR, threadCountY + extraLR, threadCountX);

    #define SRC_BUFFER(X, Y) \
        (MATRIX_ELEMENT(srcBuffer, X, Y))

    //----------------------------------------------------------------
    //
    // Read src tile
    //
    //----------------------------------------------------------------

    #define READ_SRC(X, Y) \
        devTex2D(PREP_PASTE(srcSampler, RANK), (X) + 0.5f, (Y) + 0.5f) // index to space coords

    Space readBaseX = srcBaseX - extraL;
    Space readBaseY = srcBaseY - extraL;

    ////

    COMPILE_ASSERT(threadCountX >= extraLR);
    COMPILE_ASSERT(threadCountY >= extraLR);

    bool extraX = devThreadX < extraLR;
    bool extraY = devThreadY < extraLR;

    ////

    #define READ_ITER(kX, kY) \
        SRC_BUFFER((kX) * threadCountX + devThreadX, (kY) * threadCountY + devThreadY) = \
            READ_SRC(readBaseX + (kX) * threadCountX + devThreadX, readBaseY + (kY) * threadCountY + devThreadY)

    READ_ITER(0, 0);

    if (extraX)
        READ_ITER(1, 0);

    if (extraY)
        READ_ITER(0, 1);

    if (extraX && extraY)
        READ_ITER(1, 1);

    ////

    devSyncThreads();

    //----------------------------------------------------------------
    //
    // Upsample vertically
    //
    //----------------------------------------------------------------

    devSramMatrixFor2dAccess(tmpBuffer, FloatType, threadCountX + extraLR, 2 * threadCountY, threadCountX);

    #define TMP_BUFFER(X, Y) \
        (MATRIX_ELEMENT(tmpBuffer, X, Y))

    ////

    #define UPSAMPLE_VERTICAL(bX) \
        { \
            FloatType v0 = SRC_BUFFER(bX + devThreadX, devThreadY + 0); \
            FloatType v1 = SRC_BUFFER(bX + devThreadX, devThreadY + 1); \
            FloatType v2 = SRC_BUFFER(bX + devThreadX, devThreadY + 2); \
            FloatType v3 = SRC_BUFFER(bX + devThreadX, devThreadY + 3); \
            TMP_BUFFER(bX + devThreadX, 2*devThreadY + 0) = C1*v0 + C3*v1 + C5*v2 + C7*v3; \
            TMP_BUFFER(bX + devThreadX, 2*devThreadY + 1) = C0*v0 + C2*v1 + C4*v2 + C6*v3; \
        }

    UPSAMPLE_VERTICAL(0 * threadCountX);

    if (extraX)
        UPSAMPLE_VERTICAL(1 * threadCountX);

    ////

    devSyncThreads();

    //----------------------------------------------------------------
    //
    // Upsample horizontally
    //
    //----------------------------------------------------------------

    Space bX = devThreadX;

    ////

    FloatType a0 = TMP_BUFFER(bX + 0, 2*devThreadY+0);
    FloatType a1 = TMP_BUFFER(bX + 1, 2*devThreadY+0);
    FloatType a2 = TMP_BUFFER(bX + 2, 2*devThreadY+0);
    FloatType a3 = TMP_BUFFER(bX + 3, 2*devThreadY+0);

    FloatType r00 = C1 * a0 + C3 * a1 + C5 * a2 + C7 * a3;
    FloatType r10 = C0 * a0 + C2 * a1 + C4 * a2 + C6 * a3;

    ////

    FloatType b0 = TMP_BUFFER(bX + 0, 2*devThreadY+1);
    FloatType b1 = TMP_BUFFER(bX + 1, 2*devThreadY+1);
    FloatType b2 = TMP_BUFFER(bX + 2, 2*devThreadY+1);
    FloatType b3 = TMP_BUFFER(bX + 3, 2*devThreadY+1);

    FloatType r01 = C1 * b0 + C3 * b1 + C5 * b2 + C7 * b3;
    FloatType r11 = C0 * b0 + C2 * b1 + C4 * b2 + C6 * b3;

    //----------------------------------------------------------------
    //
    // Store result
    //
    //----------------------------------------------------------------

    Space srcX = srcBaseX + devThreadX;
    Space srcY = srcBaseY + devThreadY;

    Space dstX0 = 2*srcX + 1;
    Space dstX1 = 2*srcX + 2;
    Space dstY0 = 2*srcY + 1;
    Space dstY1 = 2*srcY + 2;

    if (MATRIX_VALID_ACCESS(dst, dstX0, dstY0))
        storeNorm(MATRIX_POINTER(dst, dstX0, dstY0), r00);

    if (MATRIX_VALID_ACCESS(dst, dstX0, dstY1))
        storeNorm(MATRIX_POINTER(dst, dstX0, dstY1), r01);

    if (MATRIX_VALID_ACCESS(dst, dstX1, dstY0))
        storeNorm(MATRIX_POINTER(dst, dstX1, dstY0), r10);

    if (MATRIX_VALID_ACCESS(dst, dstX1, dstY1))
        storeNorm(MATRIX_POINTER(dst, dstX1, dstY1), r11);

}

#endif

//----------------------------------------------------------------

#if HOSTCODE

template <typename Dst>
inline const GpuKernelLink& upsampleTwiceKernelLink();

template <>
inline const GpuKernelLink& upsampleTwiceKernelLink<DST>()
    {return PREP_PASTE(upsampleTwiceKernel, DST);}

#endif
