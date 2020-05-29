#pragma once

//================================================================
//
// GPUTOOL_INDEXED_NAME
// GPUTOOL_INDEXED_SAMPLER
//
// Convenience macros.
//
//================================================================

#define GPUTOOL_INDEXED_NAME(count, Type, name) \
    PREP_FOR(count, GPT_INDEXED_NAME_FUNC, (Type, name))

#define GPT_INDEXED_NAME_FUNC(k, args) \
    GPT_INDEXED_NAME_FUNC2(k, PREP_ARG2_0 args, PREP_ARG2_1 args)

#define GPT_INDEXED_NAME_FUNC2(k, Type, name) \
    ((Type, PREP_PASTE(name, k)))

//----------------------------------------------------------------

#define GPUTOOL_INDEXED_SAMPLER(count, Type, name, interpMode, borderMode) \
    PREP_FOR(count, GPT_INDEXED_SAMPLER_FUNC, (Type, name, interpMode, borderMode))

#define GPT_INDEXED_SAMPLER_FUNC(k, args) \
    GPT_INDEXED_SAMPLER_FUNC2(k, PREP_ARG4_0 args, PREP_ARG4_1 args, PREP_ARG4_2 args, PREP_ARG4_3 args)

#define GPT_INDEXED_SAMPLER_FUNC2(k, Type, name, interpMode, borderMode) \
    ((Type, PREP_PASTE(name, k), interpMode, borderMode))
