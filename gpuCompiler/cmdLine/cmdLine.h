#pragma once

#include <vector>

#include "stlString/stlString.h"

namespace cmdLine {

using namespace std;

//================================================================
//
// parseCmdLine
//
// Can throw STL exceptions
//
//================================================================

void parseCmdLine(const CharType* cmdLineBegin, const CharType* cmdLineEnd, vector<StlString>& appendList);

inline void parseCmdLine(const StlString& cmdLine, vector<StlString>& appendList)
    {parseCmdLine(cmdLine.data(), cmdLine.data() + cmdLine.size(), appendList);}

//================================================================
//
// convertToArg
//
// Can throw STL exceptions
//
//================================================================

void convertToArg(const CharType* argBegin, const CharType* argEnd, StlString& result);

inline void convertToArg(const StlString& arg, StlString& result)
    {convertToArg(arg.data(), arg.data() + arg.size(), result);}

//----------------------------------------------------------------

}
