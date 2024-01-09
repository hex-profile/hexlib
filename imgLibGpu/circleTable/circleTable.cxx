#if HOSTCODE
#include "circleTable.h"
#endif

#include "gpuSupport/gpuTool.h"
#include "vectorTypes/vectorOperations.h"
#include "mathFuncs/gaussApprox.h"

//================================================================
//
// makeCircleTable
//
//================================================================

GPUTOOL_2D_BEG
(
    makeCircleTable,
    PREP_EMPTY,
    ((float32_x2, dst)),
    PREP_EMPTY
)
#if DEVCODE
{
    float32 r = float32(Xs) / vGlobSize.X;
    *dst = circleCcw(r);
}
#endif
GPUTOOL_2D_END

//================================================================
//
// CircleTableHolder::realloc
//
//================================================================

#if HOSTCODE

stdbool CircleTableHolder::realloc(Space size, stdPars(GpuProcessKit))
{
    require(table.realloc(point(size, 1), stdPass));
    require(makeCircleTable(table, stdPass));

    returnTrue;
}

#endif
