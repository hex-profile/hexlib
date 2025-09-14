#pragma once

//================================================================
//
// NonCopyable
//
//================================================================

struct NonCopyable
{
    inline NonCopyable() {}

    NonCopyable(const NonCopyable& that) =delete;
    NonCopyable& operator =(const NonCopyable& that) =delete;

    NonCopyable(NonCopyable&& that) =delete;
    NonCopyable& operator =(NonCopyable&& that) =delete;
};
