#include "randRange.h"

#include "numbers/float/floatType.h"

//================================================================
//
// rand<uint32>
//
//================================================================

template <>
uint32 rand(RndgenState& rndgen, uint32 lo, uint32 hi)
{
  
    uint32 result = lo;

    ////

    if (lo <= hi)
    {
        uint32 diff = hi - lo;
                              
        uint32 rem = rndgen32(rndgen);

        if (diff != uint32(0xFFFFFFFFUL)) 
            rem = rem % (diff + 1);

        result = lo + rem;
    }

    ////

    return result;

}

//================================================================
//
// rand<int32>
//
//================================================================

template <>
int32 rand(RndgenState& rndgen, int32 lo, int32 hi)
{
    int32 result = lo;

    ////

    if (lo <= hi)
    {
        uint32 diff = uint32(hi - lo); // int32 range <= uint32 range

        uint32 rem = rndgen32(rndgen);

        if (diff != uint32(0xFFFFFFFFUL)) 
            rem = rem % (diff + 1);

        result = int32(lo + rem); // result in range lo .. hi
    }

    ////

    return result;
}

//================================================================
//
// rand<float32>
//
//================================================================

template <>
float32 rand(RndgenState& rndgen, float32 lo, float32 hi)
{
    float32 result = 0;

    if (def(lo, hi) && lo <= hi)
    {
        float32 divMax32 = 0.23283064370807973754314699618685e-9f; // 1 / (2^32 - 1);
    
        float32 R = lo + float32(rndgen32(rndgen)) * divMax32 * (hi - lo);

        if (def(R))
        {
            if (R < lo) R = lo;
            if (R > hi) R = hi;
        }

        result = R;
    }

    return result;
}

//================================================================
//
// rand<float64>
//
//================================================================

template <>
float64 rand(RndgenState& rndgen, float64 lo, float64 hi)
{
    float64 result = float64Nan();

    if (def(lo, hi) && lo <= hi)
    {

        //
        // Get a number in range [0, 1], if (r1, r2) is in range [0, 2^32 - 1]
        //
        // (r1*2^(-32) + r2*2^(-64)) / (1 - 2^(-64));
        //

        float64 r1 = float64(rndgen32(rndgen));
        float64 r2 = float64(rndgen32(rndgen));

        float64 R = 
            0.23283064365386962891887177448354e-9 * r1 + 
            0.54210108624275221703311375920553e-19 * r2;

        float64 tmp = lo + R * (hi - lo);

        if (def(tmp))
        {
            if (tmp < lo) tmp = lo;
            if (tmp > hi) tmp = hi;
        }

        result = tmp;
    }

    return result;
}
