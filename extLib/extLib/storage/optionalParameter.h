#pragma once

//================================================================
//
// OptionalParameter
//
//================================================================

template <typename Type>
class OptionalParameter
{

public:

    inline OptionalParameter() =default;

    inline OptionalParameter(const Type& value)
        : exists{true}, value{value} {}

    inline explicit operator bool () const
        {return exists;}

    inline Type& operator *()
        {return *(exists ? &value : nullptr);}

    inline const Type& operator *() const
        {return *(exists ? &value : nullptr);}

private:

    bool exists = false;
    Type value;

};

//----------------------------------------------------------------

template <typename Type>
inline bool allv(const OptionalParameter<Type>& value)
    {return bool{value};}
