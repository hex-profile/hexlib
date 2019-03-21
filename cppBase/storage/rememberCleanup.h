#pragma once

//================================================================
//
// REMEMBER_CLEANUP
//
// Remembers to perform a cleanup.
// The user specifies a cleanup statement, the type and name of a variable to remember.
//
// For example:
//
// REMEMBER_CLEANUP(free(ptr));
//
// This results in calling free(ptr) at the end of the scope.
// The cleanup is always executed, even on abnormal control flow,
// like returning an error code.
//
// If you need to cancel cleanup sometimes, use REMEMBER_CLEANUP_EX.
//
//================================================================

//================================================================
//
// RC__PASTE
//
//================================================================

#define RC__PASTE_AUX(A, B) \
    A ## B

#define RC__PASTE(A, B) \
    RC__PASTE_AUX(A, B)

//================================================================
//
// RememberCleanup
//
// Modern C11 cleanup (move not implemented yet)
//
//================================================================

template <typename Action>
class RememberCleanup
{

public:

    inline RememberCleanup(const Action& action)
        : action(action) {}

    inline ~RememberCleanup()
        {action();}

private:

    Action action;

};

//----------------------------------------------------------------

template <typename Action>
inline RememberCleanup<Action> rememberCleanup(const Action& action)
    {return RememberCleanup<Action>(action);}

//----------------------------------------------------------------

#define REMEMBER_CLEANUP(action) \
    auto RC__PASTE(__cleanup, __LINE__) = rememberCleanup([&] () {action;})

//================================================================
//
// RememberCleanupEx
//
//================================================================

template <typename Action>
class RememberCleanupEx
{

public:

    inline RememberCleanupEx(const Action& action)
        : action(action) {}

    inline ~RememberCleanupEx()
        {if (active) action();}

    void cancel()
        {active = false;}

    void activate()
        {active = true;}

    void setActive(bool active)
        {this->active = active;}

private:

    bool active = true;
    Action action;

};

//----------------------------------------------------------------

template <typename Action>
inline RememberCleanupEx<Action> rememberCleanupEx(const Action& action)
    {return RememberCleanupEx<Action>(action);}

//----------------------------------------------------------------

#define REMEMBER_CLEANUP_EX(name, action) \
    auto name = rememberCleanupEx([&] () {action;})

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Outdated cleanups (before C11)
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================


//================================================================
//
// REMEMBER_CLEANUP0
//
//================================================================

#define RC__MAIN0(Name, name, statement) \
    \
    class Name \
    { \
        bool active; \
        \
    public: \
        \
        inline Name() \
            : \
            active(true) \
        { \
        } \
        \
        inline ~Name() \
        { \
            if (active) \
                {statement;} \
        } \
        \
        void cancel() {active = false;} \
        void activate() {active = true;} \
        void setActive(bool active) {this->active = active;} \
    }; \
    \
    Name name

//----------------------------------------------------------------

#define REMEMBER_CLEANUP0_EX(name, statement) \
    RC__MAIN0(RC__PASTE(name, _Type), name, statement)

#define REMEMBER_CLEANUP0(statement) \
    RC__MAIN0(RC__PASTE(__Cleanup, __LINE__), RC__PASTE(__cleanup, __LINE__), statement)

//================================================================
//
// REMEMBER_CLEANUP1
//
//================================================================

#define RC__MAIN1(Name, name, Param, param, statement) \
    \
    class Name \
    { \
        bool active; \
        Param param; \
        \
    public: \
        \
        inline Name(Param param) \
            : \
            active(true), \
            param(param) \
        { \
        } \
        \
        inline ~Name() \
        { \
            if (active) \
                {statement;} \
        } \
        \
        void cancel() {active = false;} \
        void activate() {active = true;} \
        void setActive(bool active) {this->active = active;} \
    }; \
    \
    Name name(param)

//----------------------------------------------------------------

#define REMEMBER_CLEANUP1_EX(name, statement, Param, param) \
    RC__MAIN1(RC__PASTE(name, _Type), name, Param, param, statement)

#define REMEMBER_CLEANUP1(statement, Param, param) \
    RC__MAIN1(RC__PASTE(__Cleanup, __LINE__), RC__PASTE(__cleanup, __LINE__), Param, param, statement)

//================================================================
//
// REMEMBER_CLEANUP2
//
//================================================================

#define RC__MAIN2(Name, name, Param0, param0, Param1, param1, statement) \
    \
    class Name \
    { \
        bool active; \
        Param0 param0; \
        Param1 param1; \
        \
    public: \
        \
        inline Name(Param0 param0, Param1 param1) \
            : \
            active(true), \
            param0(param0), \
            param1(param1) \
        { \
        } \
        \
        inline ~Name() \
        { \
            if (active) \
                {statement;} \
        } \
        \
        void cancel() {active = false;} \
        void activate() {active = true;} \
        void setActive(bool active) {this->active = active;} \
    }; \
    \
    Name name(param0, param1)

//----------------------------------------------------------------

#define REMEMBER_CLEANUP2_EX(name, statement, Param0, param0, Param1, param1) \
    RC__MAIN2(RC__PASTE(name, _Type), name, Param0, param0, Param1, param1, statement)

#define REMEMBER_CLEANUP2(statement, Param0, param0, Param1, param1) \
    RC__MAIN2(RC__PASTE(__Cleanup, __LINE__), RC__PASTE(__cleanup, __LINE__), Param0, param0, Param1, param1, statement)

//================================================================
//
// REMEMBER_CLEANUP3
//
//================================================================

#define RC__MAIN3(Name, name, Param0, param0, Param1, param1, Param2, param2, statement) \
    \
    class Name \
    { \
        bool active; \
        Param0 param0; \
        Param1 param1; \
        Param2 param2; \
        \
    public: \
        \
        inline Name(Param0 param0, Param1 param1, Param2 param2) \
            : \
            active(true), \
            param0(param0), \
            param1(param1), \
            param2(param2) \
        { \
        } \
        \
        inline ~Name() \
        { \
            if (active) \
                {statement;} \
        } \
        \
        void cancel() {active = false;} \
        void activate() {active = true;} \
        void setActive(bool active) {this->active = active;} \
    }; \
    \
    Name name(param0, param1, param2)

//----------------------------------------------------------------

#define REMEMBER_CLEANUP3_EX(name, statement, Param0, param0, Param1, param1, Param2, param2) \
    RC__MAIN3(RC__PASTE(name, _Type), name, Param0, param0, Param1, param1, Param2, param2, statement)

#define REMEMBER_CLEANUP3(statement, Param0, param0, Param1, param1, Param2, param2) \
    RC__MAIN3(RC__PASTE(__Cleanup, __LINE__), RC__PASTE(__cleanup, __LINE__), Param0, param0, Param1, param1, Param2, param2, statement)

//================================================================
//
// REMEMBER_CLEANUP_GENERIC
//
//================================================================

#define RC__DECL_VAR(Type, name) \
    Type name;

#define RC__DECL_PARAM(Type, name) \
    Type name,

#define RC__INIT(Type, name) \
    name(name),

#define RC__PARAM(Type, name) \
    name,

//----------------------------------------------------------------

#define RC__MAIN_GENERIC(Name, name, paramList, statement) \
    \
    class Name \
    { \
        bool active; \
        paramList(RC__DECL_VAR); \
        \
    public: \
        \
        inline Name(paramList(RC__DECL_PARAM) int=0) \
            : \
            paramList(RC__INIT) \
            active(true) \
        { \
        } \
        \
        inline ~Name() \
        { \
            if (active) \
                {statement;} \
        } \
        \
        void cancel() {active = false;} \
        void activate() {active = true;} \
        void setActive(bool active) {this->active = active;} \
    }; \
    \
    Name name(paramList(RC__PARAM) 0)

//----------------------------------------------------------------

#define REMEMBER_CLEANUP_GENERIC_EX(name, statement, paramList) \
    RC__MAIN_GENERIC(RC__PASTE(name, _Type), name, paramList, statement)

#define REMEMBER_CLEANUP_GENERIC(statement, paramList) \
    RC__MAIN_GENERIC(RC__PASTE(__Cleanup, __LINE__), RC__PASTE(__cleanup, __LINE__), paramList, statement)
