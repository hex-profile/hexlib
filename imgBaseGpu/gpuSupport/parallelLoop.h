#pragma once

#include "gpuDevice/gpuDevice.h"

//================================================================
//
// PARALLEL_LOOP_UNBASED
//
// Performs parallel "for" with 1D processor over 1D area.
//
// Loop variable is NOT adjusted for groupMember.
//
//================================================================

#define PARALLEL_LOOP_UNBASED(iterVar, iterationCount, groupMember, groupSize, iterationBody) \
    \
    { \
        constexpr auto _groupSize = (groupSize); \
        COMPILE_ASSERT(_groupSize >= 1); \
        \
        constexpr auto _iterationCount = (iterationCount); \
        COMPILE_ASSERT(_iterationCount >= 0); \
        \
        auto _groupMember = (groupMember); \
        \
        constexpr auto _iterCount = _iterationCount / _groupSize; \
        constexpr auto _iterRem = _iterationCount % _groupSize; \
        \
        bool _extra = (_iterRem) && (_groupMember < _iterRem); \
        \
        devUnrollLoop \
        \
        for (auto _n = decltype(_iterCount){0}; _n < _iterCount; ++_n) \
        { \
            const auto iterVar = _n * _groupSize; \
            {iterationBody;} \
        } \
        \
        { \
            const auto iterVar = _iterCount * _groupSize; \
            \
            if (_extra) \
                {iterationBody;} \
        } \
    }

//================================================================
//
// PARALLEL_LOOP_2D_UNBASED
//
// Loop variable is NOT adjusted for groupMember.
//
//================================================================

#define PARALLEL_LOOP_2D_UNBASED(iX, iY, iterationCountX, iterationCountY, groupMemberX, groupMemberY, groupSizeX, groupSizeY, iterationBody) \
    \
    { \
        constexpr auto _groupSizeX = (groupSizeX); \
        constexpr auto _groupSizeY = (groupSizeY); \
        \
        constexpr auto _iterCountX = (iterationCountX) / (_groupSizeX); \
        constexpr auto _iterCountY = (iterationCountY) / (_groupSizeY); \
        \
        constexpr auto iterRemX = (iterationCountX) % (_groupSizeX); \
        constexpr auto iterRemY = (iterationCountY) % (_groupSizeY); \
        \
        COMPILE_ASSERT(_iterCountX >= 0 && _iterCountY >= 0 && iterRemX >= 0 && iterRemY >= 0); /* Ensure compile-time constants */ \
        \
        auto _memberX = (groupMemberX); \
        auto _memberY = (groupMemberY); \
        \
        bool _extraX = (iterRemX) && (_memberX < iterRemX); \
        bool _extraY = (iterRemY) && (_memberY < iterRemY); \
        \
        /**/ \
        \
        devUnrollLoop \
        \
        for_count (_nY, _iterCountY) \
        { \
            const auto iY = _nY * _groupSizeY; \
            \
            devUnrollLoop \
            \
            for_count (_nX, _iterCountX) \
            { \
                const auto iX = _nX * _groupSizeX; \
                {iterationBody;} \
            } \
            \
            { \
                const auto iX = _iterCountX * _groupSizeX; \
                \
                if (_extraX) \
                    {iterationBody;} \
            } \
        } \
        \
        /**/ \
        \
        { \
            const auto iY = _iterCountY * _groupSizeY; \
            \
            devUnrollLoop \
            \
            for_count (_nX, _iterCountX) \
            { \
                const auto iX = _nX * _groupSizeX; \
                \
                if (_extraY) \
                    {iterationBody;} \
            } \
            \
            { \
                const auto iX = _iterCountX * _groupSizeX; \
                \
                if (_extraY && _extraX) \
                    {iterationBody;} \
            } \
        } \
    }

//================================================================
//
// PARALLEL_LOOP_2D_FLAT_UNROLLED
//
//================================================================

#define PARALLEL_LOOP_2D_FLAT_UNROLLED(iX, iY, groupMemberX, groupMemberY, iterationCountX, iterationCountY, groupSizeX, groupSizeY, iterationBody) \
    \
    { \
        constexpr auto _areaSizeX = (iterationCountX); \
        constexpr auto _areaSizeY = (iterationCountY); \
        \
        constexpr auto _groupSizeX = (groupSizeX); \
        constexpr auto _groupSizeY = (groupSizeY); \
        \
        constexpr auto _groupSize = _groupSizeX * _groupSizeY; \
        constexpr auto _totalArea = _areaSizeX * _areaSizeY; \
        \
        constexpr auto _iterCount = _totalArea / _groupSize; \
        constexpr auto {_iterRem = _totalArea % _groupSize; \
        \
        const auto _tid = (groupMemberX) + (groupMemberY) * _groupSizeX; \
        auto _i = _tid; \
        \
        using Index = decltype(_i); \
        using IndexU = TYPE_MAKE_UNSIGNED(Index); \
        \
        devUnrollLoop \
        for_count_ex (_n, _iterCount, _i += _groupSize) \
        { \
            const Index iY = IndexU(_i) / IndexU(_areaSizeX); /* for constant divisor, it is optimized well */ \
            const Index iX = _i - iY * _areaSizeX; \
            \
            {iterationBody;} \
        } \
        \
        if (_iterRem && _tid < _iterRem) \
        { \
            const Index iY = IndexU(_i) / IndexU(_areaSizeX); /* for constant divisor, it is optimized well */ \
            const Index iX = _i - iY * _areaSizeX; \
            \
            {iterationBody;} \
        } \
    }
