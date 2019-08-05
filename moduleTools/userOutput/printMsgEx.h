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

inline bool printMsgCheckDataProcessing(...)
    {return true;}

inline bool printMsgCheckDataProcessing(const DataProcessingKit* kit)
    {return kit->dataProcessing;}

//----------------------------------------------------------------

inline bool printMsgCheckOutputEnabled(...)
    {return true;}

inline bool printMsgCheckOutputEnabled(const OutputLevelKit* kit)
    {return kit->outputLevel >= OUTPUT_ENABLED;}

//----------------------------------------------------------------

template <typename Kit>
inline bool isOutputEnabled(const Kit& kit)
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
inline bool printMsgL(const Kit& kit, const CharArray& format, const Types&... values)
    {return !isOutputEnabled(kit) ? true : printMsg(kit.localLog, format, values...);}

template <typename Kit, typename... Types>
inline bool printMsgG(const Kit& kit, const CharArray& format, const Types&... values)
    {return !isOutputEnabled(kit) ? true : printMsg(kit.msgLog, format, values...);}

//================================================================
//
// PRINT_VAR
//
//================================================================

#define PRINT_VAR(var) \
    printMsgL(kit, STR(PREP_STRINGIZE(var) " = %0"), var)
