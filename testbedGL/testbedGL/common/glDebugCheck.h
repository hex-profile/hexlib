#pragma once

#include "stdFunc/stdFunc.h"
#include "userOutput/printMsgTrace.h"
#include "errorLog/debugBreak.h"

//================================================================
//
// CHECK_GL / REQUIRE_GL
//
//================================================================

template <typename Kit>
inline bool checkGL(const CharArray& errDesc, stdPars(Kit))
{
    bool ok = true;

    for (;;)
    {
        GLenum err = glGetError();

        if (err == GL_NO_ERROR)
            break;

        ok = false;

        printMsgTrace(kit.msgLogEx, STR("OpenGL error: %0: %1."), errDesc, (const char*) gluErrorString(err), msgErr, stdPassThru);
    }

    return ok;
}

////

#define CHECK_GL(statement) \
    ((statement), (checkGL(STR(PREP_STRINGIZE(statement)), stdPass)))

#define REQUIRE_GL(statement) \
    require(CHECK_GL(statement))

#define DEBUG_BREAK_CHECK_GL(statement) \
    ((statement), (DEBUG_BREAK_CHECK(glGetError() == GL_NO_ERROR)))

//================================================================
//
// REQUIRE_GL_FUNC
//
//================================================================

template <typename Kit>
inline bool checkGLFuncMsg(const CharArray& funcName, stdPars(Kit))
{
    return printMsgTrace(kit.msgLogEx, STR("OpenGL extension %0 is not available."), funcName, msgErr, stdPassThru);
}

////

#define REQUIRE_GL_FUNC(funcName) \
    if (funcName) ; else {checkGLFuncMsg(STR(PREP_STRINGIZE(funcName)), stdPass); returnFalse;}

////

#define REQUIRE_GL_FUNC2(f0, f1) \
    REQUIRE_GL_FUNC(f0); \
    REQUIRE_GL_FUNC(f1);
