#pragma once

#include "prepTools/prepFor.h"
#include "prepTools/prepArg.h"
#include "prepTools/prepSeq.h"
#include "prepTools/prepIncDec.h"

//================================================================
//
// List is a sequence like (A) (B) (C) (_),
// with one extra element added to support empty sets.
// The iterating macros skip the last element.
//
//================================================================

#define PREP_LIST_ELEM(i, list) \
    PREP_SEQ_ELEM(i, list)

#define PREP_LIST_SIZE(list) \
    PREP_DEC(PREP_SEQ_SIZE(list))

//================================================================
//
// PREP_LIST_FOREACH
//
// Calls specified macro for each element of the list.
//
//================================================================

#define PREP_LIST_FOREACH(list, action, extra) \
    PREP_FOR(PREP_DEC(PREP_SEQ_SIZE(list)), PREP_LIST__FOREACH0, (list, action, extra))

#define PREP_LIST__FOREACH0(n, param) \
    PREP_LIST__FOREACH1(n, PREP_ARG3_0 param, PREP_ARG3_1 param, PREP_ARG3_2 param)

#define PREP_LIST__FOREACH1(n, list, action, extra) \
    PREP_LIST__FOREACH2(action, PREP_SEQ_ELEM(n, list), extra)

#define PREP_LIST__FOREACH2(action, item, extra) \
    action(item, extra)

//================================================================
//
// PREP_LIST_FOREACH_PAIR
//
// Foreach- macro for the list of (Type, name) pairs.
//
// Calls the specified macro for each pair of the list,
// passing Type and name separately.
//
//================================================================

#define PREP_LIST_FOREACH_PAIR(list, action, extra) \
    PREP_LIST_FOREACH(list, PREP_LIST__FOREACH_PAIR0, (action, extra))

#define PREP_LIST__FOREACH_PAIR0(item, param) \
    PREP_LIST__FOREACH_PAIR1(PREP_ARG2_0 param, PREP_ARG2_1 param, PREP_ARG2_0 item, PREP_ARG2_1 item)

#define PREP_LIST__FOREACH_PAIR1(action, extra, Type, name) \
    action(Type, name, extra)

//================================================================
//
// PREP_LIST_ENUM
//
// Calls specified macro for each element of the list
// and generates comma-separated list of the results.
//
// (Currently does not support empty list)
//
//================================================================

#define PREP_LIST_ENUM(list, action, extra) \
    PREP_LIST__ENUM_MAIN(PREP_DEC(PREP_SEQ_SIZE(list)), list, action, extra)

#define PREP_LIST__ENUM_MAIN(size, list, action, extra) \
    PREP_FOR(PREP_DEC(size), PREP_LIST__ENUM_ACTION, (list, action, extra)) \
    action(PREP_SEQ_ELEM(PREP_DEC(size), list), extra)

#define PREP_LIST__ENUM_ACTION(n, param) \
    PREP_LIST__ENUM_ACTION0(n, PREP_ARG3_0 param, PREP_ARG3_1 param, PREP_ARG3_2 param)

#define PREP_LIST__ENUM_ACTION0(n, list, action, extra) \
    action(PREP_SEQ_ELEM(n, list), extra),

//================================================================
//
// PREP_LIST_ENUM_PAIR
//
// The same as PREP_LIST_ENUM, but for using with lists of pairs (Type, name).
//
//================================================================

#define PREP_LIST_ENUM_PAIR(list, action, extra) \
    PREP_LIST_ENUM(list, PREP_LIST__ENUM_PAIR__ACTION, (action, extra))

#define PREP_LIST__ENUM_PAIR__ACTION(item, param) \
    PREP_LIST__ENUM_PAIR_ACTION0(PREP_ARG2_0 param, PREP_ARG2_1 param, PREP_ARG2_0 item, PREP_ARG2_1 item)

#define PREP_LIST__ENUM_PAIR_ACTION0(action, extra, Type, name) \
    action(Type, name, extra)
