#pragma once

#if defined(_MSC_VER) && defined(_DEBUG)
    #include <crtdbg.h>
#endif

//================================================================
//
//
//
//================================================================

#if defined(_MSC_VER) && defined(_DEBUG)

using HeapState = _CrtMemState;

inline void heapStateSave(HeapState& state)
{
    _CrtMemCheckpoint(&state);
}

inline bool heapStateDifferent(const HeapState& a, const HeapState& b)
{
    HeapState difference;
    return _CrtMemDifference(&difference, &a, &b) != 0;
}

#else

using HeapState = int;

inline void heapStateSave(HeapState& state)
    {}

inline bool heapStateDifferent(const HeapState& a, const HeapState& b)
    {return false;}

#endif

//================================================================
//
// checkHeapLeaks
//
//================================================================

inline bool checkHeapLeaks()
{
    bool ok = true;

    #if defined(_MSC_VER) && defined(_DEBUG)
        ok = (_CrtDumpMemoryLeaks() == 0);
    #elif defined(_MSC_VER) && !defined(_DEBUG)
        ok = true;
    #elif defined(__GNUC__)
        ok = true;
    #else
        #error
    #endif

    return ok;
}

//================================================================
//
// checkHeapIntegrity
//
//================================================================

inline bool checkHeapIntegrity()
{
    bool ok = true;

    #if defined(_MSC_VER) && defined(_DEBUG)
        ok = (_CrtCheckMemory() != 0);
    #elif defined(_MSC_VER) && !defined(_DEBUG)
        ok = true;
    #elif defined(__GNUC__)
        ok = true;
    #else
        #error
    #endif

    return ok;
}
