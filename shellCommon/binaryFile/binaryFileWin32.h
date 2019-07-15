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

    inline bool isOpened() const {return handle != 0;}

    void close();
    stdbool open(const CharArray& filename, bool writeAccess, bool createIfNotExists, stdPars(FileDiagKit));

    stdbool truncate(stdPars(FileDiagKit));

public:

    uint64 getSize() const
        {return currentSize;}

    uint64 getPosition() const 
        {return currentPosition;}

    stdbool setPosition(uint64 pos, stdPars(FileDiagKit));

public:

    stdbool read(void* dataPtr, CpuAddrU dataSize, stdPars(FileDiagKit));
    stdbool write(const void* dataPtr, CpuAddrU dataSize, stdPars(FileDiagKit));

public:

    friend inline void exchange(BinaryFileWin32& A, BinaryFileWin32& B)
    {
        exchange(A.handle, B.handle);
        exchange(A.currentFilename, B.currentFilename);
        exchange(A.currentSize, B.currentSize);
        exchange(A.currentPosition, B.currentPosition);
    }

private:

    void* handle = 0;
    SimpleString currentFilename;

    uint64 currentSize = 0;
    uint64 currentPosition = 0; // always <= currentSize

};
