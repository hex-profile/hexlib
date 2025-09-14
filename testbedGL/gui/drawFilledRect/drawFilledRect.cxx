#include "drawFilledRect.h"

#include "gpuSupport/gpuTool.h"

#if HOSTCODE
#include "errorLog/errorLog.h"
#endif

//================================================================
//
// drawSingleRectFunc
//
//================================================================

GPUTOOL_2D_BEG
(
    drawSingleRectFunc,
    PREP_EMPTY,
    ((uint8_x4, dstImage)),
    ((Point<Space>, regionOrg))
    ((DrawSingleRectArgs, args))
)
#if DEVCODE
{
    auto pos = regionOrg + point(X, Y);

    if (allv(pos >= args.org && pos < args.end))
        *dstImage = args.color;
}
#endif
GPUTOOL_2D_END

//================================================================
//
// drawSingleRect
//
//================================================================

#if HOSTCODE

void drawSingleRect(const DrawSingleRectArgs& args, const GpuMatrix<uint8_x4>& dst, stdPars(GpuProcessKit))
{
    auto org = args.org;
    auto end = args.end;

    org = clampRange(org, point(0), dst.size());
    end = clampRange(end, point(0), dst.size());

    if_not (org < end)
        return;

    GpuMatrix<uint8_x4> dstRegion;
    REQUIRE(dst.subr(org, end, dstRegion));

    drawSingleRectFunc(dstRegion, org, args, stdPass);
}

#endif

//================================================================
//
// drawFilledRectFunc
//
//================================================================

GPUTOOL_2D_BEG
(
    drawFilledRectFunc,
    PREP_EMPTY,
    ((uint8_x4, dstImage)),
    ((Point<Space>, regionOrg))
    ((DrawFilledRectArgs, args))
)
#if DEVCODE
{
    auto pos = regionOrg + point(X, Y);

    if (allv(pos >= args.inner.org && pos < args.inner.end))
        *dstImage = args.inner.color;
    else if (allv(pos >= args.outer.org && pos < args.outer.end))
        *dstImage = args.outer.color;
}
#endif
GPUTOOL_2D_END

//================================================================
//
// drawFilledRect
//
//================================================================

#if HOSTCODE

void drawFilledRect(const DrawFilledRectArgs& args, const GpuMatrix<uint8_x4>& dst, stdPars(GpuProcessKit))
{
    auto org = minv(args.inner.org, args.outer.org);
    auto end = maxv(args.inner.end, args.outer.end);

    org = clampRange(org, point(0), dst.size());
    end = clampRange(end, point(0), dst.size());

    if_not (org < end)
        return;

    GpuMatrix<uint8_x4> dstRegion;
    REQUIRE(dst.subr(org, end, dstRegion));

    drawFilledRectFunc(dstRegion, org, args, stdPass);
}

#endif
