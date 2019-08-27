#if !(defined(PREP_ITER_DIMS) && PREP_ITER_DIMS >= 0)
    #error The dimension count needs to be defined
#endif

//----------------------------------------------------------------

#if PREP_ITER_DEPTH == 0

    #if PREP_ITER_DIMS == PREP_ITER_DEPTH
        // Do nothing
    #else
        #define PREP_ITER_FILE_0 PREP_ITERATE_ND
        #include PREP_ITERATE
    #endif

#elif PREP_ITER_DEPTH == 1

    #if PREP_ITER_DIMS == PREP_ITER_DEPTH
        #include PREP_ITER_FILE
    #else
        #define PREP_ITER_FILE_1 PREP_ITERATE_ND
        #include PREP_ITERATE
    #endif

#elif PREP_ITER_DEPTH == 2

    #if PREP_ITER_DIMS == PREP_ITER_DEPTH
        #include PREP_ITER_FILE
    #else
        #define PREP_ITER_FILE_2 PREP_ITERATE_ND
        #include PREP_ITERATE
    #endif

#elif PREP_ITER_DEPTH == 3

    #if PREP_ITER_DIMS == PREP_ITER_DEPTH
        #include PREP_ITER_FILE
    #else
        #error Unsupported range
    #endif

#else

    #error Unsupported range

#endif
