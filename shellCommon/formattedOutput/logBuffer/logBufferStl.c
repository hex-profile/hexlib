#include "logBufferStl.h"

#include "numbers/int/intType.h"

namespace logBufferStl {

//================================================================
//
// LogBufferStl::add
//
//================================================================

bool LogBufferStl::add(const CharArray& text, MsgKind kind, const TimeMoment& moment)
{
    try
    {
        msgArray.push_back(MsgRecord());

        MsgRecord& r = msgArray[msgArray.size()-1];

        r.text.assign(text.ptr, text.size);
        r.kind = kind;
        r.moment = moment;
    }
    catch (const std::exception&)
    {
        return false;
    }

    return true;
}

//================================================================
//
// restoreIndex
//
//================================================================

static inline RowInt restoreIndex(RowInt index, RowInt size)
{
    if (index <= LogBufferEnd)
        index = size + (index - LogBufferEnd);

    return clampRange(index, RowInt{0}, size);
}

//================================================================
//
// LogBufferStl::readRange
//
//================================================================

bool LogBufferStl::readRange(LogBufferReceiver& receiver, RowInt rowOrg, RowInt rowEnd)
{
    RowInt arraySize = msgArray.size();
    ensure(arraySize >= 0);

    ////

    rowOrg = restoreIndex(rowOrg, arraySize);
    rowEnd = restoreIndex(rowEnd, arraySize);

    ////

    for (RowInt i = rowOrg; i < rowEnd; ++i)
    {
        MsgRecord& r = msgArray[i];
        ensure(receiver.addRow(CharArray(r.text.data(), r.text.size()), r.kind, r.moment));
    }

    return true;
}

//----------------------------------------------------------------

}
