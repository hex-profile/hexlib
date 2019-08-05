#include "diagLogTool.h"

#include <sstream>

#include "stdFunc/stdFunc.h"
#include "formattedOutput/formatStreamStl.h"
#include "stlString/stlString.h"

//================================================================
//
// MsgLogByDiagLogStlFormatting::addMsg
//
//================================================================

bool MsgLogByDiagLogStlFormatting::addMsg(const FormatOutputAtom& v, MsgKind msgKind)
{
    using namespace std;

    if (!output)
        return true;

    try
    {
        basic_stringstream<CharType> stringStream;
        FormatStreamStlThunk formatToStream(stringStream);

        v.func(v.value, formatToStream);
        ensure(formatToStream.isOk());
        ensure(!!stringStream);

        const auto& str = stringStream.rdbuf()->str();
        output->addMsg(str.c_str(), msgKind);
    }
    catch (const exception&)
    {
        return false;
    }

    return true;
}
