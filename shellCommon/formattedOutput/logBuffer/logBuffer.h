#pragma once

#include "charType/charArray.h"
#include "userOutput/msgLog.h"
#include "kit/kit.h"
#include "userOutput/msgLog.h"
#include "timer/timer.h"

//================================================================
//
// LogBufferWriting
//
//================================================================

struct LogBufferWriting
{
    virtual bool add(const CharArray& text, MsgKind kind, const TimeMoment& moment) =0;
    virtual bool clear() =0;
};

//================================================================
//
// LogBufferReceiver
//
//================================================================

struct LogBufferReceiver
{
    virtual bool addRow(const CharArray& text, MsgKind kind, const TimeMoment& moment) =0;
};

//================================================================
//
// LogBufferEnd
//
// The constant is treated as "log buffer size", for example, read N last rows:
// readRange(LogBufferEnd - N, LogBufferEnd);
//
//================================================================

using RowInt = ptrdiff_t;

static const RowInt LogBufferEnd = -1;

//================================================================
//
// LogBufferReading
//
// Reads [rowOrg, rowEnd) range of rows.
// Clipping of the range is performed.
//
//================================================================

struct LogBufferReading
{
    virtual bool readRange(LogBufferReceiver& receiver, RowInt rowOrg, RowInt rowEnd) =0;
};

//================================================================
//
// LogBufferIO
//
//================================================================

struct LogBufferIO : public LogBufferWriting, public LogBufferReading
{
};
