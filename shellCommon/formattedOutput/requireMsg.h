#pragma once

#include "userOutput/printMsg.h"
#include "errorLog/errorLog.h"

//================================================================
//
// CHECK_MSG
//
//================================================================

#define CHECK_MSG(condition, msg) \
    CHECK_EX(condition, printMsg(kit.msgLog, msg, msgErr))

#define CHECK_MSG1(condition, msg, v0) \
    CHECK_EX(condition, printMsg(kit.msgLog, msg, v0, msgErr))

#define CHECK_MSG2(condition, msg, v0, v1) \
    CHECK_EX(condition, printMsg(kit.msgLog, msg, v0, v1, msgErr))

#define CHECK_MSG3(condition, msg, v0, v1, v2) \
    CHECK_EX(condition, printMsg(kit.msgLog, msg, v0, v1, v2, msgErr))

#define CHECK_MSG4(condition, msg, v0, v1, v2, v3) \
    CHECK_EX(condition, printMsg(kit.msgLog, msg, v0, v1, v2, v3, msgErr))

//================================================================
//
// REQUIRE_MSG
//
//================================================================

#define REQUIRE_MSG(condition, msg) \
    REQUIRE_EX(condition, printMsg(kit.msgLog, msg, msgErr))

#define REQUIRE_MSG1(condition, msg, v0) \
    REQUIRE_EX(condition, printMsg(kit.msgLog, msg, v0, msgErr))

#define REQUIRE_MSG2(condition, msg, v0, v1) \
    REQUIRE_EX(condition, printMsg(kit.msgLog, msg, v0, v1, msgErr))

#define REQUIRE_MSG3(condition, msg, v0, v1, v2) \
    REQUIRE_EX(condition, printMsg(kit.msgLog, msg, v0, v1, v2, msgErr))

#define REQUIRE_MSG4(condition, msg, v0, v1, v2, v3) \
    REQUIRE_EX(condition, printMsg(kit.msgLog, msg, v0, v1, v2, v3, msgErr))

//================================================================
//
// REQUIRE_MSG_MASKED
//
//================================================================

#define REQUIRE_MSG_MASKED(condition, msg) \
    REQUIRE_EX(condition, printMsgG(kit, msg, msgErr))

#define REQUIRE_MSG_MASKED1(condition, msg, v0) \
    REQUIRE_EX(condition, printMsgG(kit, msg, v0, msgErr))

#define REQUIRE_MSG_MASKED2(condition, msg, v0, v1) \
    REQUIRE_EX(condition, printMsgG(kit, msg, v0, v1, msgErr))

#define REQUIRE_MSG_MASKED3(condition, msg, v0, v1, v2) \
    REQUIRE_EX(condition, printMsgG(kit, msg, v0, v1, v2, msgErr))

#define REQUIRE_MSG_MASKED4(condition, msg, v0, v1, v2, v3) \
    REQUIRE_EX(condition, printMsgG(kit, msg, v0, v1, v2, v3, msgErr))
