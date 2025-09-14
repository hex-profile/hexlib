#pragma once

#include "binaryFile.h"
#include "simpleString/simpleString.h"

//================================================================
//
// BinaryFileLinux
//
//================================================================

class BinaryFileLinux : public BinaryFile
{

public:

    inline ~BinaryFileLinux() {close();}

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

    friend inline void exchange(BinaryFileLinux& A, BinaryFileLinux& B)
    {
        exchange(A.currentHandle, B.currentHandle);
        exchange(A.currentFilename, B.currentFilename);
        exchange(A.currentSize, B.currentSize);
        exchange(A.currentPosition, B.currentPosition);
    }

private:

    int currentHandle = -1;
    SimpleString currentFilename;

    uint64 currentSize = 0;
    uint64 currentPosition = 0; // always <= currentSize

};
