#pragma once

//================================================================
//
// inRange
//
//================================================================

template <typename Value, typename MinValue, typename MaxValue>
sysinline auto inRange(const Value& value, const MinValue& minValue, const MaxValue& maxValue)
{
    return minValue <= value && value <= maxValue;
}
