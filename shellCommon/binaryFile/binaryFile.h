#pragma once

#include "stdFunc/stdFunc.h"
#include "userOutput/diagnosticKit.h"
#include "storage/addrSpace.h"

//================================================================
//
// BinaryFileErrorKit
//
//================================================================

using FileDiagKit = DiagnosticKit;

//================================================================
//
// BinaryInputStream
//
//================================================================

struct BinaryInputStream
{
    virtual stdbool read(void* dataPtr, CpuAddrU dataSize, stdPars(FileDiagKit)) =0;
};

//================================================================
//
// BinaryOutputStream
//
//================================================================

struct BinaryOutputStream
{
    virtual stdbool write(const void* dataPtr, CpuAddrU dataSize, stdPars(FileDiagKit)) =0;
};

//================================================================
//
// FilePositioning
//
//================================================================

struct FilePositioning
{
    virtual uint64 getSize() const =0;
    virtual uint64 getPosition() const =0;

    virtual stdbool setPosition(uint64 pos, stdPars(FileDiagKit)) =0;
};

//================================================================
//
// FileTruncation
//
//================================================================

struct FileTruncation
{
    virtual stdbool truncate(stdPars(FileDiagKit)) =0;
};

//================================================================
//
// FileOpening
//
// The open function doesn't truncate existing file, even if 'create' flag is specified.
// Call 'truncate' explicitly if needed.
//
// If open fails, the file becomes closed.
//
//================================================================

struct FileOpening
{
    virtual bool isOpened() const =0;

    virtual stdbool open(const CharArray& filename, bool writeAccess, bool createIfNotExists, stdPars(FileDiagKit)) =0;

    virtual void close() =0;
};

//================================================================
//
// BinaryFile
//
//================================================================

struct BinaryFile : public FileOpening, public FilePositioning, public FileTruncation, public BinaryInputStream, public BinaryOutputStream {};
