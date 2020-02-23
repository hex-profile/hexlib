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

    NonCopyable(const NonCopyable&& that) =delete;
    NonCopyable& operator =(const NonCopyable&& that) =delete;
};
