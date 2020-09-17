#include "formatting/formatStream.h"

//================================================================
//
// MessageFormatter
//
//================================================================

struct MessageFormatter : public FormatOutputStream
{
    virtual void clear() =0;
    virtual bool valid() =0;
    virtual size_t size() =0;
    virtual CharType* data() =0;
};
