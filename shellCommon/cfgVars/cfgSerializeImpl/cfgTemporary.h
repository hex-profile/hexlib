#pragma once

#include "cfgVars/types/stringStorage.h"

namespace cfgVarsImpl {

//================================================================
//
// Expected limits for memory reservation.
//
//================================================================

constexpr size_t nameReserve = 100;
constexpr size_t valueReserve = 100;
constexpr size_t commentReserve = 100;
constexpr size_t blockCommentReserve = 500;

constexpr size_t filenameReserve = 0;

constexpr size_t wholeConfigReserve = 16384;

//================================================================
//
// CfgTemporary
//
//================================================================

struct CfgTemporary
{
    StringStorage name;
    StringStorage value;
    StringStorage comment;
    StringStorage blockComment;
    StringStorage tmpFilename;
    StringStorage debugStr;

    void reserve() may_throw
    {
        name.reserve(nameReserve);
        value.reserve(valueReserve);
        comment.reserve(commentReserve);
        blockComment.reserve(blockCommentReserve);
        tmpFilename.reserve(filenameReserve);
        debugStr.reserve(0);
    }

    void dealloc()
    {
        name.dealloc();
        value.dealloc();
        comment.dealloc();
        blockComment.dealloc();
        tmpFilename.dealloc();
        debugStr.dealloc();
    }
};

//----------------------------------------------------------------

}
