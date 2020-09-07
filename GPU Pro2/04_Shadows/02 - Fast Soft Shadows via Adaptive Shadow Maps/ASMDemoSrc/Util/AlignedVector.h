/* Aligned allocator for std::vector.

   Pavlo Turchyn <pavlo@nichego-novogo.net> Aug 2010 */

#ifndef __VECTOR
#define __VECTOR

#include <vector>

template<class T, size_t N> class aligned_allocator
{
public:
  typedef T value_type;
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef T& reference;
  typedef const T& const_reference;
  template <class _T> struct rebind { typedef aligned_allocator<_T, N> other; };

  template<class _T> inline aligned_allocator(const aligned_allocator<_T, N>&) throw () { }
  inline aligned_allocator() throw() { }
  inline ~aligned_allocator() throw() { }
  inline pointer adress(reference r) { return &r; }
  inline const_pointer adress(const_reference r) const { return &r; }
  inline pointer allocate(size_type n) { return (pointer)_aligned_malloc(n*sizeof(value_type), N); }
  inline void deallocate(pointer p, size_type) { _aligned_free(p); }
  inline void construct(pointer p, const value_type& x) { new (p)value_type(x); }
  inline void destroy(pointer p) { p->~value_type(); }
  inline size_type max_size() const throw() { return size_type(-1)/sizeof(value_type); }
};

template<class T, size_t N=__alignof(T)> class AlignedVector : public std::vector<T, aligned_allocator<T, N> > {};

/* std::vector implementation passes arguments by value thus making some 
   aligned types impossible to use. unaligned_blob is a wrapper that 
   allows passing around aligned types via memcpy. Sadly it works only for 
   POD types. */

template<class T> class unaligned_blob
{
public:
  finline unaligned_blob(const T& a) { memcpy(m_Data, &a, sizeof(T)); }
  finline operator T&() const { return (T&)m_Data; }

private:
  unsigned m_Data[(sizeof(T) + sizeof(unsigned) - 1)/sizeof(unsigned)];
};

template<class T, size_t N=__alignof(T)> class AlignedPODVector : public std::vector<unaligned_blob<T>, aligned_allocator<unaligned_blob<T>, N> > {};

#endif //#ifndef __VECTOR
