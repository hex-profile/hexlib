#pragma once

#include "userOutput/printMsg.h"
#include "kits/moduleKit.h"

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Provides module-level convenience thunks for printMsg function.
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// isOutputEnabled
//
//================================================================

sysinline bool printMsgCheckDataProcessing(...)
    {return true;}

sysinline bool printMsgCheckDataProcessing(const DataProcessingKit* kit)
    {return kit->dataProcessing;}

//----------------------------------------------------------------

sysinline bool printMsgCheckOutputEnabled(...)
    {return true;}

sysinline bool printMsgCheckOutputEnabled(const VerbosityKit* kit)
    {return kit->verbosity >= Verbosity::On;}

//----------------------------------------------------------------

template <typename Kit>
sysinline bool isOutputEnabled(const Kit& kit)
    {return printMsgCheckDataProcessing(&kit) && printMsgCheckOutputEnabled(&kit);}

//================================================================
//
// printMsgL
// printMsgG
//
// Convenience thunks
//
//================================================================

template <typename Kit, typename... Types>
sysinline bool printMsgL(const Kit& kit, const CharArray& format, const Types&... values)
    {return !isOutputEnabled(kit) ? true : printMsg(kit.localLog, format, values...);}

template <typename Kit, typename... Types>
sysinline bool printMsgL(const Kit& kit, CharType specialChar, const CharArray& format, const Types&... values)
    {return !isOutputEnabled(kit) ? true : printMsg(kit.localLog, specialChar, format, values...);}

//----------------------------------------------------------------

template <typename Kit, typename... Types>
sysinline bool printMsgG(const Kit& kit, const CharArray& format, const Types&... values)
    {return !isOutputEnabled(kit) ? true : printMsg(kit.msgLog, format, values...);}

template <typename Kit, typename... Types>
sysinline bool printMsgG(const Kit& kit, CharType specialChar, const CharArray& format, const Types&... values)
    {return !isOutputEnabled(kit) ? true : printMsg(kit.msgLog, specialChar, format, values...);}
