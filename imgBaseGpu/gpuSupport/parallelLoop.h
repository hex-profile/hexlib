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
        enum {_groupSizeX = (groupSizeX)}; \
        enum {_groupSizeY = (groupSizeY)}; \
        \
        enum {_iterCountX = (iterationCountX) / (_groupSizeX)}; \
        enum {_iterCountY = (iterationCountY) / (_groupSizeY)}; \
        \
        enum {iterRemX = (iterationCountX) % (_groupSizeX)}; \
        enum {iterRemY = (iterationCountY) % (_groupSizeY)}; \
        \
        COMPILE_ASSERT(_iterCountX >= 0 && _iterCountY >= 0 && iterRemX >= 0 && iterRemY >= 0); /* Ensure compile-time constants */ \
        \
        Space _memberX = (groupMemberX); \
        Space _memberY = (groupMemberY); \
        \
        bool _extraX = (iterRemX) && (_memberX < iterRemX); \
        bool _extraY = (iterRemY) && (_memberY < iterRemY); \
        \
        /**/ \
        \
        devUnrollLoop \
        \
        for (Space _nY = 0; _nY < _iterCountY; ++_nY) \
        { \
            const Space iY = _nY * _groupSizeY; \
            \
            devUnrollLoop \
            \
            for (Space _nX = 0; _nX < _iterCountX; ++_nX) \
            { \
                const Space iX = _nX * _groupSizeX; \
                {iterationBody;} \
            } \
            \
            { \
                const Space iX = _iterCountX * _groupSizeX; \
                \
                if (_extraX) \
                    {iterationBody;} \
            } \
        } \
        \
        /**/ \
        \
        { \
            const Space iY = _iterCountY * _groupSizeY; \
            \
            devUnrollLoop \
            \
            for (Space _nX = 0; _nX < _iterCountX; ++_nX) \
            { \
                const Space iX = _nX * _groupSizeX; \
                \
                if (_extraY) \
                    {iterationBody;} \
            } \
            \
            { \
                const Space iX = _iterCountX * _groupSizeX; \
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
        enum {_areaSizeX = (iterationCountX)}; \
        enum {_areaSizeY = (iterationCountY)}; \
        \
        enum {_groupSizeX = (groupSizeX)}; \
        enum {_groupSizeY = (groupSizeY)}; \
        \
        enum {_groupSize = _groupSizeX * _groupSizeY}; \
        enum {_totalArea = _areaSizeX * _areaSizeY}; \
        \
        enum {_iterCount = _totalArea / _groupSize}; \
        enum {_iterRem = _totalArea % _groupSize}; \
        \
        Space _tid = (groupMemberX) + (groupMemberY) * _groupSizeX; \
        Space _i = _tid; \
        \
        devUnrollLoop \
        for (Space _n = 0; _n < _iterCount; ++_n, _i += _groupSize) \
        { \
            Space iY = SpaceU(_i) / SpaceU(_areaSizeX); /* for constant divisor, it is optimized well */ \
            Space iX = _i - iY * _areaSizeX; \
            \
            {iterationBody;} \
        } \
        \
        if (_iterRem && _tid < _iterRem) \
        { \
            Space iY = SpaceU(_i) / SpaceU(_areaSizeX); /* for constant divisor, it is optimized well*/ \
            Space iX = _i - iY * _areaSizeX; \
            \
            {iterationBody;} \
        } \
    }
