#pragma once

#include "storage/actionHolder.h"

//================================================================
//
// CleanupHolder
//
//================================================================

template <size_t maxSize = actionHolder::defaultMaxSize>
class CleanupHolder
{

public:

    CleanupHolder() =default;

    ~CleanupHolder() {execAndClear();}

    CleanupHolder(const CleanupHolder& that) =delete;

    CleanupHolder& operator =(const CleanupHolder& that) =delete;

    CleanupHolder(CleanupHolder&& that)
    {
        cleanup = that.cleanup;
        that.cancel();
    }

    CleanupHolder& operator =(CleanupHolder&& that)
    {
        if (this != &that)
        {
            execAndClear();
            cleanup = that.cleanup;
            that.cancel();
        }

        return *this;
    }

    void execAndClear()
    {
        cleanup.execute();
        cleanup.clear();
    }

    void cancel()
    {
        cleanup.clear();
    }

    template <typename Type>
    void setCleanup(const Type& action)
    {
        execAndClear();

        static_assert(std::is_nothrow_copy_constructible<Type>::value, "");
        static_assert(std::is_nothrow_destructible<Type>::value, "");

        cleanup.setAction(action);
    }

private:

    ActionHolder<maxSize> cleanup;

};

//----------------------------------------------------------------

template <typename Type>
inline auto cleanupHolder(const Type& action)
{
    CleanupHolder<> tmp;
    tmp.setCleanup(action);
    return tmp;
}

template <size_t maxSize, typename Type>
inline auto cleanupHolder(const Type& action)
{
    CleanupHolder<maxSize> tmp;
    tmp.setCleanup(action);
    return tmp;
}
