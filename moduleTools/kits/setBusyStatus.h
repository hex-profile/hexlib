#pragma once

#include "formatting/formatModifiers.h"
#include "formatting/paramMsgOutput.h"
#include "kits/setBusyStatusKit.h"

//================================================================
//
// SetBusyStatus
//
//================================================================

struct SetBusyStatus
{
    virtual bool set(const FormatOutputAtom& message) =0;
    virtual bool reset() =0;
};

//----------------------------------------------------------------

struct SetBusyStatusNull : public SetBusyStatus
{
    virtual bool set(const FormatOutputAtom& message)
        {return true;}

    virtual bool reset()
        {return true;}
};

//================================================================
//
// setBusyStatus
//
// Printf-like function.
//
//================================================================

inline bool setBusyStatus(const SetBusyStatusKit& kit)
{
    return kit.setBusyStatus.reset();
}

//----------------------------------------------------------------

inline bool setBusyStatus(const SetBusyStatusKit& kit, const CharArray& format)
{
    return kit.setBusyStatus.set(format);
}

template <typename... Types>
inline bool setBusyStatus(const SetBusyStatusKit& kit, const CharArray& format, const Types&... values)
{
    constexpr size_t n = sizeof...(values);
    const FormatOutputAtom params[] = {values...};

    ParamMsg paramMsg(defaultSpecialChar, format, params, n);
    return kit.setBusyStatus.set(paramMsg);
}
