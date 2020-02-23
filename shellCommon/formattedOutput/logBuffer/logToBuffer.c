#include "logToBuffer.h"

#include <sstream>

#include "stdFunc/stdFunc.h"
#include "formattedOutput/formatStreamStl.h"
#include "stlString/stlString.h"

//================================================================
//
// LogToBufferThunk::addMsg
//
//================================================================

bool LogToBufferThunk::addMsg(const FormatOutputAtom& v, MsgKind msgKind)
{
    using namespace std;

    try
    {
        basic_stringstream<CharType> stringStream;
        FormatStreamStlThunk formatToStream(stringStream);

        v.func(v.value, formatToStream);
        ensure(formatToStream.valid());
        ensure(!!stringStream);

        StlString str = stringStream.rdbuf()->str();

        if (outputInterface && timer)
            outputInterface->add(CharArray(str.data(), str.size()), msgKind, timer->moment());
    }
    catch (const exception&)
    {
        return false;
    }

    return true;
}
