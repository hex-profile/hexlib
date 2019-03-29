//================================================================
//
// downsampleTwiceKernel
//
//================================================================

#if DEVCODE

template <typename Dst, typename FilterX, typename FilterY>
devDecl void PREP_PASTE(downsampleTwiceKernel_x, RANK)(const DownsampleParams<Dst, FilterX, FilterY>& o, devPars)
{

    typedef VECTOR_REBASE(Dst, float32) FloatType;

    //----------------------------------------------------------------
    //
    // Destination X, Y
    //
    //----------------------------------------------------------------

    Space dstBaseX = devGroupX * threadCountX;
    Space dstBaseY = devGroupY * threadCountY;

    //----------------------------------------------------------------
    //
    // SRAM matrix for source tile: 3 + size + 3
    //
    //----------------------------------------------------------------

    const Space extraL = 3;
    const Space extraR = 3;
    const Space extraLR = extraL + extraR;

    const Space srcSramSizeX = 2 * threadCountX + extraLR;
    const Space srcSramSizeY = 2 * threadCountY + extraLR;

    devSramMatrixFor2dAccess(srcBuffer, FloatType, srcSramSizeX, srcSramSizeY, threadCountX);

    #define SRC_BUFFER(X, Y) \
        (*MATRIX_POINTER(srcBuffer, X, Y))

    //----------------------------------------------------------------
    //
    // Read src tile
    //
    //----------------------------------------------------------------

    #define READ_SRC(X, Y) \
        devTex2D(PREP_PASTE(srcSampler, RANK), (X) + 0.5f, (Y) + 0.5f) // index to space coords

    Space srcBaseX = 2 * dstBaseX - extraL + o.srcOfs.X;
    Space srcBaseY = 2 * dstBaseY - extraL + o.srcOfs.Y;

    ////

    COMPILE_ASSERT(threadCountX >= extraLR);
    COMPILE_ASSERT(threadCountY >= extraLR);

    bool extraX = devThreadX < extraLR;
    bool extraY = devThreadY < extraLR;

    ////

    #define READ_ITER(kX, kY) \
        SRC_BUFFER((kX) * threadCountX + devThreadX, (kY) * threadCountY + devThreadY) = \
            READ_SRC(srcBaseX + (kX) * threadCountX + devThreadX, srcBaseY + (kY) * threadCountY + devThreadY)

    READ_ITER(0, 0); READ_ITER(1, 0);
    READ_ITER(0, 1); READ_ITER(1, 1);

    if (extraX) 
    {
        READ_ITER(2, 0); 
        READ_ITER(2, 1);
    }

    if (extraY)
    {
        READ_ITER(0, 2); 
        READ_ITER(1, 2);
    }

    if (extraX && extraY)
        READ_ITER(2, 2);

    ////

    devSyncThreads();

    //----------------------------------------------------------------
    //
    // Downsample Y
    //
    //----------------------------------------------------------------

    devSramMatrixFor2dAccess(tmpBuffer, FloatType, srcSramSizeX, threadCountY, threadCountX);

    #define TMP_BUFFER(X, Y) \
        (*MATRIX_POINTER(tmpBuffer, X, Y))

    ////

    Space bY = 2 * devThreadY;

    #define DOWNSAMPLE_VERTICAL(bX) \
        TMP_BUFFER(bX, devThreadY) = \
            FilterY::C0() * helpRead(SRC_BUFFER(bX, bY + 0)) + \
            FilterY::C1() * helpRead(SRC_BUFFER(bX, bY + 1)) + \
            FilterY::C2() * helpRead(SRC_BUFFER(bX, bY + 2)) + \
            FilterY::C3() * helpRead(SRC_BUFFER(bX, bY + 3)) + \
            FilterY::C4() * helpRead(SRC_BUFFER(bX, bY + 4)) + \
            FilterY::C5() * helpRead(SRC_BUFFER(bX, bY + 5)) + \
            FilterY::C6() * helpRead(SRC_BUFFER(bX, bY + 6)) + \
            FilterY::C7() * helpRead(SRC_BUFFER(bX, bY + 7));

    DOWNSAMPLE_VERTICAL(devThreadX + 0 * threadCountX);
    DOWNSAMPLE_VERTICAL(devThreadX + 1 * threadCountX);

    if (extraX)
    DOWNSAMPLE_VERTICAL(devThreadX + 2 * threadCountX);

    ////

    devSyncThreads();

    //----------------------------------------------------------------
    //
    // Downsample X
    //
    //----------------------------------------------------------------

    Space bX = 2 * devThreadX;

    FloatType result = 
        FilterX::C0() * helpRead(TMP_BUFFER(bX + 0, devThreadY)) +
        FilterX::C1() * helpRead(TMP_BUFFER(bX + 1, devThreadY)) +
        FilterX::C2() * helpRead(TMP_BUFFER(bX + 2, devThreadY)) +
        FilterX::C3() * helpRead(TMP_BUFFER(bX + 3, devThreadY)) +
        FilterX::C4() * helpRead(TMP_BUFFER(bX + 4, devThreadY)) +
        FilterX::C5() * helpRead(TMP_BUFFER(bX + 5, devThreadY)) +
        FilterX::C6() * helpRead(TMP_BUFFER(bX + 6, devThreadY)) +
        FilterX::C7() * helpRead(TMP_BUFFER(bX + 7, devThreadY));

    //----------------------------------------------------------------
    //
    // Write output
    //
    //----------------------------------------------------------------

    MATRIX_EXPOSE_EX(o.dst, dst);

    Space dstX = dstBaseX + devThreadX;
    Space dstY = dstBaseY + devThreadY;

    if (dstX < dstSizeX && dstY < dstSizeY)
        storeNorm(MATRIX_POINTER(dst, dstX, dstY), result);
}

#endif
