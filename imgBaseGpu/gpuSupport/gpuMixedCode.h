#pragma once

//================================================================
//
// HOSTCODE
// DEVCODE
//
// Just for syntax highlighting
//
//================================================================

#ifndef HOSTCODE
#define HOSTCODE 1
#endif

#ifndef DEVCODE
#define DEVCODE 1
#endif

//================================================================
//
// DEV_ONLY
//
//================================================================

#if DEVCODE
    #define DEV_ONLY(code) code
#else
    #define DEV_ONLY(code)
#endif

//================================================================
//
// HOST_ONLY
//
//================================================================

#if HOSTCODE
    #define HOST_ONLY(code) code
#else
    #define HOST_ONLY(code)
#endif
