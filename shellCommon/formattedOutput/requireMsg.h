#pragma once

#include "userOutput/printMsg.h"
#include "errorLog/errorLog.h"

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
