#include "imageProviderMemcpy.h"

#include "errorLog/errorLog.h"

//================================================================
//
// ImageProviderMemcpy::saveBgr32
//
//================================================================

stdbool ImageProviderMemcpy::saveBgr32(const MatrixAP<Pixel>& dest, stdNullPars)
{
    REQUIRE(source.size() == dest.size());

    MATRIX_EXPOSE(source);
    MATRIX_EXPOSE(dest);

    auto sourceRow = sourceMemPtr;
    auto destRow = destMemPtr;

    for_count (Y, sourceSizeY)
    {
        memcpy(unsafePtr(destRow, sourceSizeX), unsafePtr(sourceRow, sourceSizeX), sourceSizeX * sizeof(Pixel));
        destRow += destMemPitch;
        sourceRow += sourceMemPitch;
    }

    returnTrue;
}
