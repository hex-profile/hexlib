#include "bmpFile.h"

#include "vectorTypes/vectorBase.h"
#include "errorLog/errorLog.h"
#include "userOutput/errorLogEx.h"
#include "data/spacex.h"

namespace bmpFile {

//================================================================
//
// BmpWriter::clearMemory
//
//================================================================

void BmpWriter::clearMemory()
{
    if (memoryPtr)
    {
        free(memoryPtr);
        memoryPtr = nullptr;
        memorySize = 0;
    }
}

//================================================================
//
// bmpAlignment
//
//================================================================

static const Space bmpAlignmentMask = 3;

//================================================================
//
// getAlignedPitch
//
//================================================================

template <typename Pixel>
stdbool getAlignedPitch(Space sizeX, Space& pitch, stdPars(ErrorLogKit))
{
    REQUIRE(sizeX >= 0);

    REQUIRE(sizeX <= spaceMax / Space(sizeof(Pixel)));
    Space rowMemSize = sizeX * Space(sizeof(Pixel));

    REQUIRE(rowMemSize <= spaceMax - bmpAlignmentMask);
    Space rowAlignedSize = (rowMemSize + bmpAlignmentMask) & (~bmpAlignmentMask);

    Space bufSizeX = SpaceU(rowAlignedSize) / SpaceU(sizeof(Pixel));
    REQUIRE(bufSizeX * Space(sizeof(Pixel)) == rowAlignedSize);

    pitch = bufSizeX;

    returnTrue;
}

//================================================================
//
// BmpWriter::write
//
//================================================================

template <typename Pixel>
stdbool BmpWriter::writeFunc(const Matrix<const Pixel>& image, const CharArray& filename, stdPars(Kit))
{

    MATRIX_EXPOSE(image);

    //----------------------------------------------------------------
    //
    // Desired pitch.
    //
    //----------------------------------------------------------------

    Space desiredPitch = 0;
    require(getAlignedPitch<Pixel>(imageSizeX, desiredPitch, stdPass));

    //----------------------------------------------------------------
    //
    // Convert image if pitch doesn't satisfy BMP requirements.
    //
    //----------------------------------------------------------------

    auto usedImage = image;

    ////

    if_not (image.memPitch() == desiredPitch)
    {
        REQUIRE(desiredPitch >= imageSizeX);

        Space bufferArea = 0;
        REQUIRE(safeMul(desiredPitch, imageSizeY, bufferArea));

        Space requiredSize = 0;
        REQUIRE(safeMul(bufferArea, Space(sizeof(Pixel)), requiredSize));

        if_not (size_t(requiredSize) <= this->memorySize)
        {
            auto newPtr = realloc(memoryPtr, clampMin(requiredSize, 1));
            REQUIRE_TRACE(newPtr, STR("Cannot allocate buffer memory"));

            memoryPtr = newPtr;
            memorySize = requiredSize;
        }

        ////

        Matrix<Pixel> buffer{(Pixel*) memoryPtr, desiredPitch, imageSizeX, imageSizeY, MatrixValidityAssertion{}};
        MATRIX_EXPOSE(buffer);

        auto imagePtr = imageMemPtr;
        auto bufferPtr = bufferMemPtr;

        for (Space Y = 0; Y < imageSizeY; ++Y)
        {
            memcpy(bufferPtr, imagePtr, imageSizeX * sizeof(Pixel));
            imagePtr += imageMemPitch;
            bufferPtr += bufferMemPitch;
        }

        ////

        usedImage = buffer;
    }

    //----------------------------------------------------------------
    //
    //
    //
    //----------------------------------------------------------------

    REQUIRE(usedImage.memPitch() == desiredPitch);

    //----------------------------------------------------------------
    //
    //
    //
    //----------------------------------------------------------------




    ////

    returnTrue;
}

//================================================================
//
// BmpWriter::write
//
//================================================================

stdbool BmpWriter::write(const Matrix<const uint8>& image, const CharArray& filename, stdPars(Kit))
    {return writeFunc(image, filename, stdPass);}

stdbool BmpWriter::write(const Matrix<const uint8_x4>& image, const CharArray& filename, stdPars(Kit))
    {return writeFunc(image, filename, stdPass);}

//----------------------------------------------------------------

}
