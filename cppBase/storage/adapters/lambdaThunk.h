#pragma once

#include "compileTools/compileTools.h"

//================================================================
//
// LAMBDA_THUNK
//
// Makes an adapter from a lambda to the specified virtual interface.
//
// Extended versions support virtual interfaces with N functions
// using N lambdas.
//
// Lambda is stored BY VALUE, in some use cases
// a lambda temporary is passed to constructor.
//
//================================================================

#define LAMBDA_THUNK(thunk, Api, funcProto, funcBody) \
    \
    template <typename Lambda> \
    class Api##Thunk : public Api \
    { \
        \
    public: \
        \
        Api##Thunk(const Lambda& lambda) \
            : lambda{lambda} {} \
        \
        virtual funcProto \
            {return funcBody;} \
        \
    private: \
        \
        Lambda lambda; \
        \
    }; \
    \
    struct Api##ThunkMaker \
    { \
        template <typename Lambda> \
        sysinline auto operator |(const Lambda& lambda) const \
        { \
            return Api##Thunk<Lambda>(lambda); \
        } \
    }; \
    \
    constexpr Api##ThunkMaker thunk;

//================================================================
//
// LAMBDA_THUNK2
//
//================================================================

#define LAMBDA_THUNK2(api, Api, funcProto0, funcBody0, funcProto1, funcBody1) \
    \
    template <typename Lambda0, typename Lambda1> \
    class Api##Thunk : public Api \
    { \
        \
    public: \
        \
        Api##Thunk(const Lambda0& lambda0, const Lambda1& lambda1) \
            : lambda0{lambda0}, lambda1{lambda1} {} \
        \
        virtual funcProto0 \
            {return funcBody0;} \
        \
        virtual funcProto1 \
            {return funcBody1;} \
        \
    private: \
        \
        Lambda0 lambda0; \
        Lambda1 lambda1; \
        \
    }; \
    \
    template <typename Lambda0, typename Lambda1> \
    sysinline auto api(const Lambda0& lambda0, const Lambda1& lambda1) \
        {return Api##Thunk<Lambda0, Lambda1>{lambda0, lambda1};}

//================================================================
//
// LAMBDA_THUNK3
//
//================================================================

#define LAMBDA_THUNK3(api, Api, funcProto0, funcBody0, funcProto1, funcBody1, funcProto2, funcBody2) \
    \
    template <typename Lambda0, typename Lambda1, typename Lambda2> \
    class Api##Thunk : public Api \
    { \
        \
    public: \
        \
        Api##Thunk(const Lambda0& lambda0, const Lambda1& lambda1, const Lambda2& lambda2) \
            : lambda0{lambda0}, lambda1{lambda1}, lambda2{lambda2} {} \
        \
        virtual funcProto0 \
            {return funcBody0;} \
        \
        virtual funcProto1 \
            {return funcBody1;} \
        \
        virtual funcProto2 \
            {return funcBody2;} \
        \
    private: \
        \
        Lambda0 lambda0; \
        Lambda1 lambda1; \
        Lambda2 lambda2; \
        \
    }; \
    \
    template <typename Lambda0, typename Lambda1, typename Lambda2> \
    sysinline auto api(const Lambda0& lambda0, const Lambda1& lambda1, const Lambda2& lambda2) \
        {return Api##Thunk<Lambda0, Lambda1, Lambda2>{lambda0, lambda1, lambda2};}

//================================================================
//
// LAMBDA_THUNK4
//
//================================================================

#define LAMBDA_THUNK4(api, Api, funcProto0, funcBody0, funcProto1, funcBody1, funcProto2, funcBody2, funcProto3, funcBody3) \
    \
    template <typename Lambda0, typename Lambda1, typename Lambda2, typename Lambda3> \
    class Api##Thunk : public Api \
    { \
        \
    public: \
        \
        Api##Thunk(const Lambda0& lambda0, const Lambda1& lambda1, const Lambda2& lambda2, const Lambda3& lambda3) \
            : lambda0{lambda0}, lambda1{lambda1}, lambda2{lambda2}, lambda3{lambda3} {} \
        \
        virtual funcProto0 \
            {return funcBody0;} \
        \
        virtual funcProto1 \
            {return funcBody1;} \
        \
        virtual funcProto2 \
            {return funcBody2;} \
        \
        virtual funcProto3 \
            {return funcBody3;} \
        \
    private: \
        \
        Lambda0 lambda0; \
        Lambda1 lambda1; \
        Lambda2 lambda2; \
        Lambda3 lambda3; \
        \
    }; \
    \
    template <typename Lambda0, typename Lambda1, typename Lambda2, typename Lambda3> \
    sysinline auto api(const Lambda0& lambda0, const Lambda1& lambda1, const Lambda2& lambda2, const Lambda3& lambda3) \
        {return Api##Thunk<Lambda0, Lambda1, Lambda2, Lambda3>{lambda0, lambda1, lambda2, lambda3};}
