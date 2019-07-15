//================================================================
//
// convertKernel
//
//================================================================

#if DEVCODE

devDefineKernel(PREP_PASTE(convertKernel, SUFFIX), ConvertParamsYuvBgr<DST_PIXEL>, o)
{
    MATRIX_EXPOSE_EX(o.dst, dst);

    Point<Space> dstChromaBase = devGroupIdx * threadCount() - 1;

    //----------------------------------------------------------------
    //
    // SRAM matrix for source tile: 1 + size + 2
    //
    //----------------------------------------------------------------

    const Space extraL = 1;
    const Space extraR = 2;
    const Space extraLR = extraL + extraR;

    devSramMatrixFor2dAccess(srcBufferU, float32, threadCountX + extraLR, threadCountY + extraLR, threadCountX);
    devSramMatrixFor2dAccess(srcBufferV, float32, threadCountX + extraLR, threadCountY + extraLR, threadCountX);

    #define SRC_BUFFERU(X, Y) (MATRIX_ELEMENT(srcBufferU, X, Y))
    #define SRC_BUFFERV(X, Y) (MATRIX_ELEMENT(srcBufferV, X, Y))

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

    Point<float32> chromaReadBaseTex = convertIndexToPos(dstChromaBase + o.srcOffsetDiv2 - extraL) * o.chromaTexstep;

    #define READ_ITER(bX, bY, chromaReader) \
        { \
            auto chromaPos = chromaReadBaseTex + convertFloat32(point(bX, bY) + devThreadIdx) * o.chromaTexstep; \
            float32_x2 value = chromaReader(chromaPos); \
            SRC_BUFFERU((bX) + devThreadX, (bY) + devThreadY) = value.x; \
            SRC_BUFFERV((bX) + devThreadX, (bY) + devThreadY) = value.y; \
        }

    #define READ_EVERYTHING(chromaReader) \
        { \
            READ_ITER(0 * threadCountX, 0 * threadCountY, chromaReader); \
            \
            if (extraX) \
                READ_ITER(1 * threadCountX, 0 * threadCountY, chromaReader); \
            \
            if (extraY) \
                READ_ITER(0 * threadCountX, 1 * threadCountY, chromaReader); \
            \
            if (extraX && extraY) \
                READ_ITER(1 * threadCountX, 1 * threadCountY, chromaReader); \
        }

    ////

    #define CHROMA_PACKED_READER(ofs) \
        tex2D(chromaSamplerPacked, ofs);

    #define CHROMA_PLANAR_READER(ofs) \
        make_float32_x2(tex2D(chromaSamplerU, ofs), tex2D(chromaSamplerV, ofs));

    if (o.chromaIsPacked)
        READ_EVERYTHING(CHROMA_PACKED_READER)
    else
        READ_EVERYTHING(CHROMA_PLANAR_READER)

    ////

    devSyncThreads();

    //----------------------------------------------------------------
    //
    // Upsample vertically
    //
    //----------------------------------------------------------------

    devSramMatrixDense(tmpBufferU, float32, threadCountX + extraLR, 2 * threadCountY);
    devSramMatrixDense(tmpBufferV, float32, threadCountX + extraLR, 2 * threadCountY);

    #define TMP_BUFFERU(X, Y) (MATRIX_ELEMENT(tmpBufferU, X, Y))
    #define TMP_BUFFERV(X, Y) (MATRIX_ELEMENT(tmpBufferV, X, Y))

    ////

    #define UPSAMPLE_VERTICAL(srcBuffer, dstBuffer, bX) \
        { \
            float32 v0 = srcBuffer((bX) + devThreadX, devThreadY + 0); \
            float32 v1 = srcBuffer((bX) + devThreadX, devThreadY + 1); \
            float32 v2 = srcBuffer((bX) + devThreadX, devThreadY + 2); \
            float32 v3 = srcBuffer((bX) + devThreadX, devThreadY + 3); \
            dstBuffer((bX) + devThreadX, 2*devThreadY + 0) = C1*v0 + C3*v1 + C5*v2 + C7*v3; \
            dstBuffer((bX) + devThreadX, 2*devThreadY + 1) = C0*v0 + C2*v1 + C4*v2 + C6*v3; \
        } 

    UPSAMPLE_VERTICAL(SRC_BUFFERU, TMP_BUFFERU, 0 * threadCountX);
    UPSAMPLE_VERTICAL(SRC_BUFFERV, TMP_BUFFERV, 0 * threadCountX);

    if (extraX)
    {
        UPSAMPLE_VERTICAL(SRC_BUFFERU, TMP_BUFFERU, 1 * threadCountX);
        UPSAMPLE_VERTICAL(SRC_BUFFERV, TMP_BUFFERV, 1 * threadCountX);
    }

    ////

    devSyncThreads();

    //----------------------------------------------------------------
    //
    // Upsample horizontally
    //
    //----------------------------------------------------------------

    #define UPSAMPLE_HORIZONTAL(tmpBuffer, Y, r0, r1) \
        { \
            float32 v0 = tmpBuffer(devThreadX + 0, Y); \
            float32 v1 = tmpBuffer(devThreadX + 1, Y); \
            float32 v2 = tmpBuffer(devThreadX + 2, Y); \
            float32 v3 = tmpBuffer(devThreadX + 3, Y); \
            \
            r0 = C1 * v0 + C3 * v1 + C5 * v2 + C7 * v3; \
            r1 = C0 * v0 + C2 * v1 + C4 * v2 + C6 * v3; \
        }

    float32 u00, u01, u10, u11;
    UPSAMPLE_HORIZONTAL(TMP_BUFFERU, 2*devThreadY + 0, u00, u10)
    UPSAMPLE_HORIZONTAL(TMP_BUFFERU, 2*devThreadY + 1, u01, u11)

    float32 v00, v01, v10, v11;
    UPSAMPLE_HORIZONTAL(TMP_BUFFERV, 2*devThreadY + 0, v00, v10)
    UPSAMPLE_HORIZONTAL(TMP_BUFFERV, 2*devThreadY + 1, v01, v11)

    ////

    Point<Space> dstChromaPos = dstChromaBase + devThreadIdx;

    Space dstX0 = 2*dstChromaPos.X + 1;
    Space dstX1 = 2*dstChromaPos.X + 2;
    Space dstY0 = 2*dstChromaPos.Y + 1;
    Space dstY1 = 2*dstChromaPos.Y + 2;

    Point<Space> srcOffset = 2 * o.srcOffsetDiv2;
    Space srcX0 = dstX0 + srcOffset.X;
    Space srcX1 = dstX1 + srcOffset.X;
    Space srcY0 = dstY0 + srcOffset.Y;
    Space srcY1 = dstY1 + srcOffset.Y;

    float32 lumaX0 = convertIndexToPos(srcX0) * o.lumaTexstep.X;
    float32 lumaX1 = convertIndexToPos(srcX1) * o.lumaTexstep.X;
    float32 lumaY0 = convertIndexToPos(srcY0) * o.lumaTexstep.Y;
    float32 lumaY1 = convertIndexToPos(srcY1) * o.lumaTexstep.Y;

    float32 y00 = devTex2D(lumaSampler, lumaX0, lumaY0);
    float32 y01 = devTex2D(lumaSampler, lumaX0, lumaY1);
    float32 y10 = devTex2D(lumaSampler, lumaX1, lumaY0);
    float32 y11 = devTex2D(lumaSampler, lumaX1, lumaY1);

    ////

    #if CONVERT == CONVERT_RGB

        float32_x4 r00 = convertYPbPrToBgr(y00, u00, v00);
        float32_x4 r01 = convertYPbPrToBgr(y01, u01, v01);
        float32_x4 r10 = convertYPbPrToBgr(y10, u10, v10);
        float32_x4 r11 = convertYPbPrToBgr(y11, u11, v11);

    #elif CONVERT == CONVERT_YUV

        float32_x4 r00 = make_float32_x4(y00, u00, v00, 0);
        float32_x4 r01 = make_float32_x4(y01, u01, v01, 0);
        float32_x4 r10 = make_float32_x4(y10, u10, v10, 0);
        float32_x4 r11 = make_float32_x4(y11, u11, v11, 0);

    #else

        #error

    #endif

    //----------------------------------------------------------------
    //
    // Convert
    //
    //----------------------------------------------------------------

    DST_PIXEL t00 = convertNormClamp<DST_PIXEL>(r00);
    DST_PIXEL t01 = convertNormClamp<DST_PIXEL>(r01);
    DST_PIXEL t10 = convertNormClamp<DST_PIXEL>(r10);
    DST_PIXEL t11 = convertNormClamp<DST_PIXEL>(r11);

    //----------------------------------------------------------------
    //
    // Valid area
    //
    //----------------------------------------------------------------

    bool sensibleX0 = SpaceU(srcX0) < SpaceU(o.srcSize.X);
    bool sensibleX1 = SpaceU(srcX1) < SpaceU(o.srcSize.X);
    bool sensibleY0 = SpaceU(srcY0) < SpaceU(o.srcSize.Y);
    bool sensibleY1 = SpaceU(srcY1) < SpaceU(o.srcSize.Y);

    if_not (sensibleX0 && sensibleY0) t00 = o.outerColor;
    if_not (sensibleX0 && sensibleY1) t01 = o.outerColor;
    if_not (sensibleX1 && sensibleY0) t10 = o.outerColor;
    if_not (sensibleX1 && sensibleY1) t11 = o.outerColor;

    //----------------------------------------------------------------
    //
    // Store result
    //
    //----------------------------------------------------------------

    if (MATRIX_VALID_ACCESS(dst, dstX0, dstY0))
        MATRIX_ELEMENT(dst, dstX0, dstY0) = t00;

    if (MATRIX_VALID_ACCESS(dst, dstX0, dstY1))
        MATRIX_ELEMENT(dst, dstX0, dstY1) = t01;

    if (MATRIX_VALID_ACCESS(dst, dstX1, dstY0))
        MATRIX_ELEMENT(dst, dstX1, dstY0) = t10;

    if (MATRIX_VALID_ACCESS(dst, dstX1, dstY1))
        MATRIX_ELEMENT(dst, dstX1, dstY1) = t11;
}

#endif

//================================================================
//
// convertYuv420ToBgr
//
//================================================================

#if HOSTCODE

template <typename SrcPixel, typename SrcPixel2, typename DST_PIXEL>
stdbool PREP_PASTE(convertYuv420To, SUFFIX)
(
    const GpuMatrix<const SrcPixel>& srcLuma, 
    const GpuMatrix<const SrcPixel2>& srcChromaPacked,
    const GpuMatrix<const SrcPixel>& srcChromaU,
    const GpuMatrix<const SrcPixel>& srcChromaV,
    const Point<Space>& srcOffset,
    const DST_PIXEL& outerColor,
    const GpuMatrix<DST_PIXEL>& dst, 
    stdPars(GpuProcessKit)
)
{
    stdScopedBegin;

    if_not (kit.dataProcessing)
        returnTrue;

    stdEnterElemCount(areaOf(dst.size() >> 1));

    ////

    bool chromaIsPacked = hasData(srcChromaPacked);

    Point<Space> chromaSize{};

    if (chromaIsPacked)
        chromaSize = srcChromaPacked.size();
    else
    {
        REQUIRE(equalSize(srcChromaU, srcChromaV));
        chromaSize = srcChromaU.size();
    }

    ////

    REQUIRE(chromaSize >= srcLuma.size() / 2);
    REQUIRE(srcOffset % 2 == 0);

    //----------------------------------------------------------------
    //
    // The kernel covers destination 2k+1, 2k+2.
    // Let dst index be 0..N-1.
    //
    // 2k+1 <= 0 | 2k <= -1 | k <= -1/2 | k <= -1
    // 2k+2 >= N-1 | 2k >= N-3 | k >= (N-3)/2 | k >= ceil((N-3)/2) == ceil((N+1)/2) - 2
    //
    // So, k range is [-1, ceil((N+1)/2) - 2], or [-1, ceil((N+1)/2) - 1)
    //
    //----------------------------------------------------------------

    Point<Space> kernOrg = point(-1);
    Point<Space> kernEnd = dst.size() >> 1;

    //
    // Below checks should always be positive
    //

    REQUIRE(2*kernOrg+1 <= 0);

    REQUIRE(2*(kernEnd-1)+2 >= dst.size()-1);
    REQUIRE(2*(kernEnd-2)+2 < dst.size()-1);

    Point<Space> kernRange = kernEnd - kernOrg;

    //----------------------------------------------------------------
    //
    // 
    //
    //----------------------------------------------------------------

    require(kit.gpuSamplerSetting.setSamplerImage(lumaSampler, srcLuma, BORDER_MIRROR, false, true, true, stdPass));

    ////

    if (chromaIsPacked)
        require(kit.gpuSamplerSetting.setSamplerImage(chromaSamplerPacked, srcChromaPacked, BORDER_MIRROR, false, true, true, stdPass));
    else
    {
        require(kit.gpuSamplerSetting.setSamplerImage(chromaSamplerU, srcChromaU, BORDER_MIRROR, false, true, true, stdPass));
        require(kit.gpuSamplerSetting.setSamplerImage(chromaSamplerV, srcChromaV, BORDER_MIRROR, false, true, true, stdPass));
    }

    ////

    ConvertParamsYuvBgr<DST_PIXEL> params
    {
        computeTexstep(srcLuma),
        chromaIsPacked,
        computeTexstep(chromaSize),
        srcOffset / 2, srcLuma.size(), 
        outerColor, 
        dst
    };

    require
    (
        kit.gpuKernelCalling.callKernel
        (
            divUpNonneg(kernRange, point(threadCountX, threadCountY)),
            point(threadCountX, threadCountY),
            areaOf(dst),
            PREP_PASTE(convertKernel, SUFFIX),
            params,
            kit.gpuCurrentStream,
            stdPass
        )
    );

    ////

    stdScopedEnd;
}

#endif
