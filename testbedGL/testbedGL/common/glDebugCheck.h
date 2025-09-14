#pragma once

#include "stdFunc/stdFunc.h"
#include "userOutput/printMsgTrace.h"
#include "errorLog/debugBreak.h"

//================================================================
//
// REQUIRE_GL
//
//================================================================

template <typename Kit>
sysinline void checkGL(const CharArray& errDesc, stdPars(Kit))
{
    bool ok = true;

    for (; ;)
    {
        GLenum err = glGetError();

        if (err == GL_NO_ERROR)
            break;

        ok = false;

        printMsgTrace(STR("OpenGL error: %0: %1."), errDesc, (const char*) gluErrorString(err), msgErr, stdPassThru);
    }

    require(ok);
}

////

#define REQUIRE_GL(statement) \
    do { \
        {statement;} \
        checkGL(STR(PREP_STRINGIZE(statement)), stdPass); \
    } while (0)

////

#define DEBUG_BREAK_CHECK_GL(statement) \
    ((statement), (DEBUG_BREAK_CHECK(glGetError() == GL_NO_ERROR)))

#define REMEMBER_CLEANUP_GL(statement) \
    REMEMBER_CLEANUP(DEBUG_BREAK_CHECK_GL(statement))

//================================================================
//
// REQUIRE_GL_FUNC
//
//================================================================

template <typename Kit>
sysinline void checkGLFuncMsg(const CharArray& funcName, stdPars(Kit))
{
    printMsgTrace(STR("OpenGL extension %0 is not available."), funcName, msgErr, stdPassThru);
}

////

#define REQUIRE_GL_FUNC(funcName) \
    if (funcName) ; else {checkGLFuncMsg(STR(PREP_STRINGIZE(funcName)), stdPass); returnFalse;}

////

#define REQUIRE_GL_FUNC2(f0, f1) \
    REQUIRE_GL_FUNC(f0); \
    REQUIRE_GL_FUNC(f1);
