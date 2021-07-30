#pragma once

#include "prepTools/prepSeq.h"

//================================================================
//
// Convenient "foreach" for sequences like (a) (b) (c).
//
//================================================================

//================================================================
//
// PREP_SEQ_FOREACH
//
//================================================================

#define PREP_SEQ_FOREACH(seq, action, extra) \
    PREP_FOR(PREP_SEQ_SIZE(seq), PREP_SEQ_FOREACH__ITER0, (seq, action, extra))

#define PREP_SEQ_FOREACH__ITER0(i, args) \
    PREP_SEQ_FOREACH__ITER1(i, PREP_ARG3_0 args, PREP_ARG3_1 args, PREP_ARG3_2 args)

#define PREP_SEQ_FOREACH__ITER1(i, seq, action, extra) \
    PREP_SEQ_FOREACH__ITER2(PREP_SEQ_ELEM(i, seq), action, extra)

#define PREP_SEQ_FOREACH__ITER2(elem, action, extra) \
    action(elem, extra)

//================================================================
//
// PREP_SEQ_FOREACH_PAIR
//
//================================================================

#define PREP_SEQ_FOREACH_PAIR(seq, action, extra) \
    PREP_FOR(PREP_SEQ_SIZE(seq), PREP_SEQ_FOREACH_PAIR__ITER0, (seq, action, extra))

#define PREP_SEQ_FOREACH_PAIR__ITER0(i, args) \
    PREP_SEQ_FOREACH_PAIR__ITER1(i, PREP_ARG3_0 args, PREP_ARG3_1 args, PREP_ARG3_2 args)

#define PREP_SEQ_FOREACH_PAIR__ITER1(i, seq, action, extra) \
    PREP_SEQ_FOREACH_PAIR__ITER2(PREP_SEQ_ELEM(i, seq), action, extra)

#define PREP_SEQ_FOREACH_PAIR__ITER2(elem, action, extra) \
    PREP_SEQ_FOREACH_PAIR__ITER3(PREP_ARG2_0 elem, PREP_ARG2_1 elem, action, extra)

#define PREP_SEQ_FOREACH_PAIR__ITER3(v0, v1, action, extra) \
    action(v0, v1, extra)

//================================================================
//
// PREP_SEQ_FOREACH_TRIPLET
//
//================================================================

#define PREP_SEQ_FOREACH_TRIPLET(seq, action, extra) \
    PREP_FOR(PREP_SEQ_SIZE(seq), PREP_SEQ_FOREACH_TRIPLET__ITER0, (seq, action, extra))

#define PREP_SEQ_FOREACH_TRIPLET__ITER0(i, args) \
    PREP_SEQ_FOREACH_TRIPLET__ITER1(i, PREP_ARG3_0 args, PREP_ARG3_1 args, PREP_ARG3_2 args)

#define PREP_SEQ_FOREACH_TRIPLET__ITER1(i, seq, action, extra) \
    PREP_SEQ_FOREACH_TRIPLET__ITER2(PREP_SEQ_ELEM(i, seq), action, extra)

#define PREP_SEQ_FOREACH_TRIPLET__ITER2(elem, action, extra) \
    PREP_SEQ_FOREACH_TRIPLET__ITER3(PREP_ARG3_0 elem, PREP_ARG3_1 elem, PREP_ARG3_2 elem, action, extra)

#define PREP_SEQ_FOREACH_TRIPLET__ITER3(v0, v1, v2, action, extra) \
    action(v0, v1, v2, extra)
