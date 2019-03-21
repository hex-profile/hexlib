#pragma once

//================================================================
//
// DestructibleInterface
//
//================================================================

template <typename Interface>
struct DestructibleInterface : public Interface
{
    virtual ~DestructibleInterface() {}
};
