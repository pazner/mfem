// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_HANDLE_HPP
#define MFEM_HANDLE_HPP

#include "../config/config.hpp"
#include <memory>

namespace mfem
{

namespace internal
{

/// @brief An internal class used in the implementation of Handle.
///
/// MaybeOwningPtr is a smart pointer class that can represent either an owned
/// or not-owned pointed; this is configurable at runtime. The type of the
/// pointer is erased.
class MaybeOwningPtr
{
protected:
   void *ptr;
   bool owns_ptr;
   void(*deleter)(void *);
   template <typename T>
   static void Delete(void *ptr)
   {
      T *t = static_cast<T*>(ptr);
      delete t;
   }
public:
   template <typename T>
   MaybeOwningPtr(T *ptr_, bool owns_ptr_)
      : ptr(ptr_), owns_ptr(owns_ptr_), deleter(&Delete<T>) { }
   template <typename T>
   MaybeOwningPtr(const T *ptr_, bool owns_ptr_)
      : MaybeOwningPtr(const_cast<T*>(ptr_), owns_ptr_) { }
   MaybeOwningPtr(const MaybeOwningPtr &) = delete;
   MaybeOwningPtr(MaybeOwningPtr &&) = delete;
   MaybeOwningPtr &operator=(const MaybeOwningPtr &) = delete;
   MaybeOwningPtr &operator=(MaybeOwningPtr &&other) = delete;
   void SetOwner(bool owns) { owns_ptr = owns; }
   template <typename T> T *Get() const { return static_cast<T*>(ptr); }
   bool IsOwner() const { return owns_ptr; }
   ~MaybeOwningPtr() { if (owns_ptr) { deleter(ptr); } }
};

}

/// @brief A smart pointer class that may represent either shared ownership, or
/// a non-owning borrow.
///
/// A Handle may either be owning or non-owning. Non-owning Handle%s point to
/// externally owned data; it is the responsibility of the user to ensure that
/// the data remains valid as long as the Handle is alive. The underlying data
/// will be valid as long as there is at least one live copy of the Handle.
///
/// Both types of Handle%s can be copied, moved, stored in standard containers,
/// etc. Handle uses MaybeOwningPtr to implement owning and non-owning semantics
/// and <a href="https://en.cppreference.com/w/cpp/memory/shared_ptr">
/// std::shared_ptr</a> for reference counting.
///
/// A non-owning Handle may assume ownership over the pointed-to object, and an
/// owning Handle maybe release ownership. Note that both of these operations
/// apply to all existing copies of the Handle. If there are multiple live
/// non-owning copies of a Handle, and one of them assumes ownership of the
/// pointer, then they all become owning Handle%s, and vice-versa.
template <typename T>
class Handle
{
   using MaybeOwningPtr = internal::MaybeOwningPtr;

   /// Pointer to the object.
   std::shared_ptr<MaybeOwningPtr> ptr;

   /// @brief Types @a Handle<T> and @a %Handle\<U\> are friends to allow
   /// construction of one from another when @a T and @a U are convertible types
   /// (i.e. creation of a Handle to a base class from a Handle to a derived
   /// class).
   template <typename U> friend class Handle;
public:
   /// Create an empty (null) Handle.
   Handle() = default;

   /// @brief Create a Handle pointing to @a t
   ///
   /// If @a take_ownership is true, then the Handle assumes ownership over the
   /// pointer, and it should not be deleted externally. Otherwise, the Handle
   /// will be non-owning, and it is the user's responsibility to ensure the
   /// correct lifetime of @a t.
   Handle(T *t, bool take_ownership = false)
   {
      ptr = std::make_shared<MaybeOwningPtr>(t, take_ownership);
   }

   /// @brief Constructs a copy of @a u, where type @a U is convertible to @a T.
   ///
   /// This allows the construction of Handle<Base> from Handle<Derived>.
   template <typename U> Handle(const Handle<U> &u) : ptr(u.ptr) { }

   /// @brief Move-constructs from @a u, where type @a U is convertible to @a T.
   ///
   /// See @ref Handle(const Handle<U>&).
   template <typename U> Handle(Handle<U> &&u) : ptr(std::move(u.ptr)) { }

   /// @brief Copy constructor.
   ///
   /// Copying an owning Handle results in another owning handle. Copying a
   /// non-owning handle results in a non-owning handle.
   Handle(const Handle &other) = default;

   /// Move constructor (see Handle(const Handle&)).
   Handle(Handle &&other) = default;

   /// Copy assignment (see Handle(const Handle&)).
   Handle &operator=(const Handle &other) = default;

   /// Move assignment (see Handle(const Handle&)).
   Handle &operator=(Handle &&other) = default;

   /// Destructor. If the Handle is owning, decrement the reference count.
   ~Handle() = default;

   /// Returns the contained pointer (may be null).
   T *Get() const { return ptr ? ptr->Get<T>() : nullptr; }

   /// Dereference operator. The Handle must be non-null.
   T &operator*() const { return *Get(); }

   /// Member access (arrow) operator. The Handle must be non-null.
   T *operator->() const { return Get(); }

   /// @brief Returns true if the Handle is owning, false if it is non-owning.
   ///
   /// Returns false if the Handle is null (empty).
   bool IsOwner() const { return ptr ? ptr->IsOwner() : false; }

   /// Returns true if the Handle is non-null.
   explicit operator bool() const { return Get() != nullptr; }

   /// Assumes or releases ownership of the data.
   void SetOwner(bool owns) { if (ptr) { ptr->SetOwner(owns); } }

   /// @brief Reset the Handle to be empty.
   ///
   /// If the Handle is owning, this will decrement the reference count.
   void Reset() { ptr.reset(); }

   /// @brief Reset the Handle to point to @a t.
   ///
   /// Optionally, the Handle may assume ownership of the pointer (see @ref
   /// Handle(T*, bool)).
   void Reset(T *t, bool take_ownership = false)
   {
      ptr = std::make_shared<MaybeOwningPtr>(t, take_ownership);
   }

   /// Return a new owning Handle pointing to @a t.
   static Handle<T> Owning(T *t) { return Handle<T>(t, true); }

   /// Return a new non-owning Handle pointing to @a t.
   static Handle<T> NonOwning(T *t) { return Handle<T>(t, false); }

   /// Return a new owning Handle, constructed using the given arguments.
   template <typename... Args>
   static Handle<T> MakeOwning(Args&&... args)
   {
      T *t = new T(std::forward<Args>(args)...);
      return Handle<T>(t, true);
   }
};

} // namespace mfem

#endif
