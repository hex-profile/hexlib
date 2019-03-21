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

template <typename Kit>
inline bool printMsgL(const Kit& kit, const CharArray& format, MsgKind msgKind = msgInfo)
    {return !isOutputEnabled(kit) ? true : printMsg(kit.localLog, format, msgKind);}

template <typename Kit>
inline bool printMsgG(const Kit& kit, const CharArray& format, MsgKind msgKind = msgInfo)
    {return !isOutputEnabled(kit) ? true : printMsg(kit.msgLog, format, msgKind);}

//----------------------------------------------------------------

#define PRINTUSR__PRINT_MSG(n, _) \
    \
    template <PREP_ENUM_INDEXED(n, typename T), typename Kit> \
    inline bool printMsgL(const Kit& kit, const CharArray& format, PREP_ENUM_INDEXED_PAIR(n, const T, &v), MsgKind msgKind = msgInfo) \
        {return !isOutputEnabled(kit) ? true : printMsg(kit.localLog, format, PREP_ENUM_INDEXED(n, v), msgKind);} \
    \
    template <PREP_ENUM_INDEXED(n, typename T), typename Kit> \
    inline bool printMsgG(const Kit& kit, const CharArray& format, PREP_ENUM_INDEXED_PAIR(n, const T, &v), MsgKind msgKind = msgInfo) \
        {return !isOutputEnabled(kit) ? true : printMsg(kit.msgLog, format, PREP_ENUM_INDEXED(n, v), msgKind);}

#define PRINTUSR__PRINT_MSG_THUNK(n, _) \
    PRINTUSR__PRINT_MSG(PREP_INC(n), _)

PREP_FOR1(PRINTMSG__MAX_COUNT, PRINTUSR__PRINT_MSG_THUNK, _)

//================================================================
//
// PRINT_VAR
//
//================================================================

#define PRINT_VAR(var) \
    printMsgL(kit, STR(PREP_STRINGIZE(var) " = %0"), var)
