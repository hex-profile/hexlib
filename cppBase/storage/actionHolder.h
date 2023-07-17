#pragma once

#include <type_traits>
#include <stddef.h>

namespace actionHolder {

//================================================================
//
// Action interface
//
//================================================================

struct Action
{
    virtual void execute() =0;
    virtual Action* constructCopy(void* memory) =0;
    virtual ~Action() {}
};

//================================================================
//
// ActionImpl
//
//================================================================

template <typename Type>
class ActionImpl : public Action
{

public:

    ActionImpl(const Type& action)
        :
        action(action)
    {
    }

    void execute()
    {
        action();
    }

    Action* constructCopy(void* memory)
    {
        return new (memory) ActionImpl(*this);
    }

private:

    Type action;

};

//================================================================
//
// defaultMaxSize
//
//================================================================

constexpr size_t defaultMaxSize = 16;

//================================================================
//
// ActionHolder
//
// Holds arbitrary action.
//
//================================================================

template <size_t maxSize = defaultMaxSize>
class ActionHolder
{

    using Self = ActionHolder<maxSize>;

public:

    ActionHolder() =default;

    ActionHolder(const Self& that)
    {
        if (that.action)
            action = that.action->constructCopy(&storage);
    }

    Self& operator =(const Self& that)
    {
        if (this != &that)
        {
            clear();

            if (that.action)
                action = that.action->constructCopy(&storage);
        }

        return *this;
    }

    ~ActionHolder()
    {
        clear();
    }

public:

    template <typename Type>
    void setAction(const Type& action)
    {
        clear();

        using Impl = ActionImpl<Type>;

        static_assert(sizeof(Impl) <= sizeof(storage), "");
        static_assert(alignof(Impl) <= alignof(decltype(storage)), "");

        this->action = new (&storage) Impl(action);
    }

public:

    void clear()
    {
        if (action)
        {
            action->~Action();
            action = nullptr;
        }
    }

    void execute()
    {
        if (action)
            action->execute();
    }

    bool hasAction() const
    {
        return action != nullptr;
    }

private:

    Action* action = nullptr;

    std::aligned_storage_t<maxSize> storage;

};

//----------------------------------------------------------------

}

using actionHolder::ActionHolder;
