#include "imageProviderMemcpy.h"

#include "errorLog/errorLog.h"

//================================================================
//
// ImageProviderMemcpy::saveImage
//
//================================================================

stdbool ImageProviderMemcpy::saveImage(const Matrix<Pixel>& dest, stdNullPars)
{
    REQUIRE(source.size() == dest.size());

    MATRIX_EXPOSE(source);
    MATRIX_EXPOSE(dest);

    auto sourceRow = sourceMemPtr;
    auto destRow = destMemPtr;

    for (Space Y = 0; Y < sourceSizeY; ++Y)
    {
        memcpy(unsafePtr(destRow, sourceSizeX), unsafePtr(sourceRow, sourceSizeX), sourceSizeX * sizeof(Pixel));
        destRow += destMemPitch;
        sourceRow += sourceMemPitch;
    }

    returnTrue;
}
