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

GPUTOOL_2D
(
    makeCircleTable,
    PREP_EMPTY,
    ((float32_x2, dst)),
    PREP_EMPTY,
    {
        float32 r = float32(Xs) / vGlobSize.X;
        *dst = circleCcw(r);
    }
)

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
