//================================================================
//
// convertKernel
//
//================================================================

#if DEVCODE

devDefineKernel(PREP_PASTE3(convertKernel, DST_PIXEL, DST_PIXEL2), PREP_PASS2(ConvertBgrNvParams<DST_PIXEL, DST_PIXEL2>), o)
{

    //----------------------------------------------------------------
    //
    // Prepare
    //
    //----------------------------------------------------------------

    MATRIX_EXPOSE_EX(o.src, src);
    MATRIX_EXPOSE_EX(o.dstLuma, dstLuma);

    ////

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

    devSramMatrixDense(srcBufferU, float32, srcSramSizeX, srcSramSizeY);
    devSramMatrixDense(srcBufferV, float32, srcSramSizeX, srcSramSizeY);

    #define SRC_BUFFER_U(X, Y) \
        (*MATRIX_POINTER(srcBufferU, X, Y))

    #define SRC_BUFFER_V(X, Y) \
        (*MATRIX_POINTER(srcBufferV, X, Y))

    //----------------------------------------------------------------
    //
    // Read src tile
    //
    //----------------------------------------------------------------

    COMPILE_ASSERT(threadCountX >= extraLR);
    COMPILE_ASSERT(threadCountY >= extraLR);

    bool extraX = devThreadX < extraLR;
    bool extraY = devThreadY < extraLR;

    ////

    Space srcBaseX = 2 * dstBaseX - extraL + devThreadX;
    Space srcBaseY = 2 * dstBaseY - extraL + devThreadY;

    srcBaseY += o.verticalSourceOffset;

    float32 srcBaseXs = srcBaseX + 0.5f;
    float32 srcBaseYs = srcBaseY + 0.5f;

    ////

    #define READ_ITER(kX, kY) \
        \
        { \
            const Space tX = (kX) * threadCountX; \
            const Space tY = (kY) * threadCountY; \
            \
            float32_x4 bgrValue = devTex2D(srcSampler, srcBaseXs + tX, srcBaseYs + tY); \
            \
            float32 Yf, Pb, Pr; \
            convertBgrToYPbPr(bgrValue, Yf, Pb, Pr); \
            \
            SRC_BUFFER_U(tX + devThreadX, tY + devThreadY) = Pb; \
            SRC_BUFFER_V(tX + devThreadX, tY + devThreadY) = Pr; \
            \
            Space srcX = srcBaseX + tX; \
            Space srcY = srcBaseY + tY; \
            \
            if (MATRIX_VALID_ACCESS(dstLuma, srcX, srcY)) \
                *MATRIX_POINTER(dstLuma, srcX, srcY) = \
                    convertNormClamp<DST_PIXEL>(Yf); \
        }

    ////

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

    Space bY = 2 * devThreadY;

    #define DOWNSAMPLE_VERTICAL(bX, srcBuffer) \
        ( \
            C0 * helpRead(srcBuffer(bX, bY + 0)) + \
            C1 * helpRead(srcBuffer(bX, bY + 1)) + \
            C2 * helpRead(srcBuffer(bX, bY + 2)) + \
            C3 * helpRead(srcBuffer(bX, bY + 3)) + \
            C4 * helpRead(srcBuffer(bX, bY + 4)) + \
            C5 * helpRead(srcBuffer(bX, bY + 5)) + \
            C6 * helpRead(srcBuffer(bX, bY + 6)) + \
            C7 * helpRead(srcBuffer(bX, bY + 7)) \
        )

    ////

    #define DOWNSAMPLE_VERTICAL_ALL(srcBuffer) \
        { \
            float32 vertValue0 = DOWNSAMPLE_VERTICAL(devThreadX + 0 * threadCountX, srcBuffer); \
            float32 vertValue1 = DOWNSAMPLE_VERTICAL(devThreadX + 1 * threadCountX, srcBuffer); \
            float32 vertValue2 = 0; \
            if (extraX) vertValue2 = DOWNSAMPLE_VERTICAL(devThreadX + 2 * threadCountX, srcBuffer); \
            devSyncThreads(); \
            \
            srcBuffer(devThreadX + 0 * threadCountX, devThreadY) = vertValue0; \
            srcBuffer(devThreadX + 1 * threadCountX, devThreadY) = vertValue1; \
            if (extraX) \
            srcBuffer(devThreadX + 2 * threadCountX, devThreadY) = vertValue2; \
            devSyncThreads(); \
        }

    DOWNSAMPLE_VERTICAL_ALL(SRC_BUFFER_U)
    DOWNSAMPLE_VERTICAL_ALL(SRC_BUFFER_V)

    //----------------------------------------------------------------
    //
    // Downsample X
    //
    //----------------------------------------------------------------

    Space bX = 2 * devThreadX;

    #define DOWNSAMPLE_HORIZONTAL(bX, srcBuffer) \
        ( \
            C0 * helpRead(srcBuffer(bX + 0, devThreadY)) + \
            C1 * helpRead(srcBuffer(bX + 1, devThreadY)) + \
            C2 * helpRead(srcBuffer(bX + 2, devThreadY)) + \
            C3 * helpRead(srcBuffer(bX + 3, devThreadY)) + \
            C4 * helpRead(srcBuffer(bX + 4, devThreadY)) + \
            C5 * helpRead(srcBuffer(bX + 5, devThreadY)) + \
            C6 * helpRead(srcBuffer(bX + 6, devThreadY)) + \
            C7 * helpRead(srcBuffer(bX + 7, devThreadY)) \
        )

    float32 resultU = DOWNSAMPLE_HORIZONTAL(bX, SRC_BUFFER_U);
    float32 resultV = DOWNSAMPLE_HORIZONTAL(bX, SRC_BUFFER_V);

    //----------------------------------------------------------------
    //
    // Write output
    //
    //----------------------------------------------------------------

    MATRIX_EXPOSE_EX(o.dstChroma, dstChroma);

    ////

    Space chromaSizeX = dstChromaSizeX;
    Space chromaSizeY = dstChromaSizeY;

    ////

    Space dstX = dstBaseX + devThreadX;
    Space dstY = dstBaseY + devThreadY;

    if (dstX < chromaSizeX && dstY < chromaSizeY)
    {
        float32_x2 chromaValue = make_float32_x2(resultU, resultV);
        *MATRIX_POINTER(dstChroma, dstX, dstY) = convertNormClamp<DST_PIXEL2>(chromaValue);
    }
}

#endif 

//================================================================
//
// convertBgr32ToYuv420
//
//================================================================

#if HOSTCODE

template <>
bool convertBgr32ToYuv420<DST_PIXEL, DST_PIXEL2>
(
    const GpuMatrix<const uint8_x4>& src,
    const GpuMatrix<DST_PIXEL>& dstLuma,
    const GpuMatrix<DST_PIXEL2>& dstChroma,
    stdPars(GpuProcessKit)
)
{
    stdBegin;

    if_not (kit.dataProcessing)
        return true; // no allocation

    ////

    REQUIRE(equalSize(src, dstLuma));
    Point<Space> lumaSize = src.size();

    ////

    Point<Space> chromaSize = dstChroma.size();

    ////

    Point<Space> lumaHalfUp = (lumaSize + 1) >> 1;
    Point<Space> lumaHalfDn = lumaSize >> 1;

    REQUIRE(chromaSize == lumaHalfUp || chromaSize == lumaHalfDn);

    Point<Space> domainSize = maxv(chromaSize, lumaHalfUp);

    //
    // Compute memory space array, covering the matrix.
    // Correct the array base address if the pitch is negative.
    //

    GpuMatrix<const uint8_x4> tmpSrc = src;
    GpuMatrix<DST_PIXEL> tmpDstLuma = dstLuma;
    GpuMatrix<DST_PIXEL2> tmpDstChroma = dstChroma;

    Space verticalSourceOffset = 0;

    if (src.memPitch() < 0)
    {
        verticalSourceOffset = lumaSize.Y - 2*chromaSize.Y;

        tmpSrc = flipMatrix(tmpSrc); 
        tmpDstLuma = flipMatrix(tmpDstLuma);
        tmpDstChroma = flipMatrix(tmpDstChroma);
    }

    ////

    MATRIX_EXPOSE(tmpSrc);
    REQUIRE(tmpSrcMemPitch >= 0);

    ////

    require
    (
        kit.gpuSamplerSetting.setSamplerImage
        (
            srcSampler,
            tmpSrc,
            BORDER_CLAMP,
            false,
            true,
            false,
            stdPass
        )
    );

    ////

    ConvertBgrNvParams<DST_PIXEL, DST_PIXEL2> params;
    params.src = tmpSrc;
    params.verticalSourceOffset = verticalSourceOffset;
    params.dstLuma = tmpDstLuma;
    params.dstChroma = tmpDstChroma;

    require
    (
        kit.gpuKernelCalling.callKernel
        (
            divUpNonneg(domainSize, point(threadCountX, threadCountY)),
            point(threadCountX, threadCountY),
            areaOf(dstChroma),
            PREP_PASTE3(convertKernel, DST_PIXEL, DST_PIXEL2),
            params,
            kit.gpuCurrentStream,
            stdPass
        )
    );

    ////

    stdEnd;
}

#endif
