#pragma once

//================================================================
//
// DIV_DOWN_NONNEG
// DIV_UP_NONNEG
// DIV_NEAREST_NONNEG
//
//================================================================

#define DIV_DOWN_NONNEG(A, B) \
    ((A) / (B))

#define DIV_UP_NONNEG(A, B) \
    (((A) + ((B) - 1)) / (B))

#define DIV_NEAREST_NONNEG(A, B) \
    (((A) + ((B) >> 1)) / (B))

//================================================================
//
// ALIGN_UP_NONNEG
// ALIGN_DOWN_NONNEG
//
//================================================================

#define ALIGN_UP_NONNEG(a, b) \
    (DIV_UP_NONNEG(a, b) * (b))

#define ALIGN_DOWN_NONNEG(a, b) \
    (DIV_DOWN_NONNEG(a, b) * (b))
