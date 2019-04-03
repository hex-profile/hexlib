#pragma once

//================================================================
//
// ObjectHolder usage examples
//
//================================================================

#if 0

// Create empty object holder
ObjectHolder<MyInterface> holder;

// Create specific content object.
holder.create<MyImpl>();

// Re-create different content object. Old content object is destructed and deallocated.
holder.create<MyImpl2>();

// Destruct and deallocate content object.
// (The holder's destructor destructs and deallocates content object automatically.)
holder.clear();

// Get interface pointer. The pointer can be NULL.
MyInterface* ptr = holder;

// Use object holder as interface.
holder->func();

#endif

//================================================================
//
// ObjectHolder
//
// Placeholder of specific instance of abstract interface.
//
//================================================================

template <typename Interface>
class ObjectHolder
{

public:

    ObjectHolder() {ptr = 0; deleter = 0;}
    ~ObjectHolder() {clear();}

public:

    inline void clear()
    {
        if (ptr != 0)
        {
            deleter(ptr);
            ptr = 0;
            deleter = 0;
        }
    }

public:

    template <typename Implementation>
    inline void create()
    {
        clear();
        ptr = new (std::nothrow) Implementation;
        deleter = deleteHelper<Implementation>;
    }

public:

    operator Interface* () const {return ptr;}
    Interface* operator () () const {return ptr;}
    Interface* operator -> () const {return ptr;}

private:

    template <typename Implementation>
    static void deleteHelper(Interface* ptr)
        {delete (Implementation*) ptr;}

    typedef void Deleter(Interface* ptr);

private:

    Interface* ptr;
    Deleter* deleter;

};
