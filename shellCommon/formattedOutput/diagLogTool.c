#include "diagLogTool.h"

#include <sstream>

#include "stdFunc/stdFunc.h"
#include "formattedOutput/formatStreamStl.h"
#include "stlString/stlString.h"

//================================================================
//
// DiagLogMsgLog::addMsg
//
//================================================================

bool DiagLogMsgLog::addMsg(const FormatOutputAtom& v, MsgKind msgKind)
{
    using namespace std;

    if (!output)
        return true;

    try
    {
        basic_stringstream<CharType> stringStream;
        FormatStreamStlThunk formatToStream(stringStream);

        v.func(v.value, formatToStream);
        require(formatToStream.isOk());
        require(!!stringStream);

        const auto& str = stringStream.rdbuf()->str();
        output->add(str.data(), str.size(), msgKind);
    }
    catch (const exception&)
    {
        require(false);
    }

    return true;
}
