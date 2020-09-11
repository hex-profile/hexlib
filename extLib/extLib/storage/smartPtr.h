#pragma once

#ifndef HEXLIB_SMART_PTR
#define HEXLIB_SMART_PTR

#include <memory>
#include <utility>

//================================================================
//
// Just synonyms for std:: pointers, behaviour is exactly the same.
//
//================================================================

//================================================================
//
// UniquePtr
//
//================================================================

template <typename Type>
using UniquePtr = std::unique_ptr<Type>;

//----------------------------------------------------------------

template <typename Type, typename... Args>
inline UniquePtr<Type> makeUnique(Args&&... args)
{
    return std::make_unique<Type>(std::forward<Args>(args)...);
}

//================================================================
//
// SharedPtr
//
//================================================================

template <typename Type>
using SharedPtr = std::shared_ptr<Type>;

template <typename Type>
using WeakPtr = std::weak_ptr<Type>;

//----------------------------------------------------------------

template <typename Type, typename... Args>
inline SharedPtr<Type> makeShared(Args&&... args)
{
    return std::make_shared<Type>(std::forward<Args>(args)...);
}

//----------------------------------------------------------------

#endif // HEXLIB_SMART_PTR
