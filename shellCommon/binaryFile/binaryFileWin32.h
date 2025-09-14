#pragma once

#include "binaryFile.h"
#include "simpleString/simpleString.h"

//================================================================
//
// BinaryFileWin32
//
//================================================================

class BinaryFileWin32 : public BinaryFile
{

public:

    inline ~BinaryFileWin32() {close();}

    inline bool isOpened() const {return currentHandle != 0;}

    void close();
    void open(const CharArray& filename, bool writeAccess, bool createIfNotExists, stdPars(FileDiagKit));

    void truncate(stdPars(FileDiagKit));

public:

    uint64 getSize() const
        {return currentSize;}

    uint64 getPosition() const
        {return currentPosition;}

    void setPosition(uint64 pos, stdPars(FileDiagKit));

public:

    void read(void* dataPtr, CpuAddrU dataSize, stdPars(FileDiagKit));
    void write(const void* dataPtr, CpuAddrU dataSize, stdPars(FileDiagKit));

public:

    friend inline void exchange(BinaryFileWin32& A, BinaryFileWin32& B)
    {
        exchange(A.currentHandle, B.currentHandle);
        exchange(A.currentFilename, B.currentFilename);
        exchangeByCopying(A.currentSize, B.currentSize);
        exchangeByCopying(A.currentPosition, B.currentPosition);
    }

private:

    void* currentHandle = 0;
    SimpleString currentFilename;

    uint64 currentSize = 0;
    uint64 currentPosition = 0; // always <= currentSize

};
