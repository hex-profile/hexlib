#pragma once

#include <vector>

#include "formattedOutput/logBuffer/logBuffer.h"
#include "stlString/stlString.h"

namespace logBufferStl {

//================================================================
//
// MsgRecord
//
//================================================================

struct MsgRecord
{
    StlString text;
    MsgKind kind;
    TimeMoment moment;
};

//================================================================
//
// MsgArray
//
//================================================================

using MsgArray = std::vector<MsgRecord>;

//================================================================
//
// MsgArray
//
//================================================================

class LogBufferStl : public LogBufferIO
{

public:

    bool add(const CharArray& text, MsgKind kind, const TimeMoment& moment);

    bool clear()
    {
        msgArray.clear();
        return true;
    }

public:

    bool readRange(LogBufferReceiver& receiver, RowInt rowOrg, RowInt rowEnd);

private:

    MsgArray msgArray;

};

//----------------------------------------------------------------

}

using logBufferStl::LogBufferStl;
