/*
    Copyright 2005-2008 Intel Corporation.  All Rights Reserved.

    This file is part of Threading Building Blocks.

    Threading Building Blocks is free software; you can redistribute it
    and/or modify it under the terms of the GNU General Public License
    version 2 as published by the Free Software Foundation.

    Threading Building Blocks is distributed in the hope that it will be
    useful, but WITHOUT ANY WARRANTY; without even the implied warranty
    of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Threading Building Blocks; if not, write to the Free Software
    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

    As a special exception, you may use this file as part of a free software
    library without restriction.  Specifically, if other files instantiate
    templates or use macros or inline functions from this file, or you compile
    this file and link it with other files to produce an executable, this
    file does not by itself cause the resulting executable to be covered by
    the GNU General Public License.  This exception does not however
    invalidate any other reasons why the executable file might be covered by
    the GNU General Public License.
*/

#ifndef __TBB_machine_H
#error Do not include this file directly; include tbb_machine.h instead
#endif

#include <windows.h>

#if defined(__INTEL_COMPILER)
#define __TBB_fence_for_acquire() __asm { __asm nop }
#define __TBB_fence_for_release() __asm { __asm nop }
#elif _MSC_VER >= 1300
extern "C" void _ReadWriteBarrier();
#pragma intrinsic(_ReadWriteBarrier)
#define __TBB_fence_for_acquire() _ReadWriteBarrier()
#define __TBB_fence_for_release() _ReadWriteBarrier()
#endif

#define __TBB_WORDSIZE 4
#define __TBB_BIG_ENDIAN 0

#if defined(_MSC_VER) && defined(_Wp64)
    // Workaround for overzealous compiler warnings in /Wp64 mode
    #pragma warning (push)
    #pragma warning (disable: 4244 4267)
#endif /* _MSC_VER && _Wp64 */

extern "C" {
    __int64 __TBB_machine_cmpswp8 (volatile void *ptr, __int64 value, __int64 comparand );
    __int64 __TBB_machine_fetchadd8 (volatile void *ptr, __int64 addend );
    __int64 __TBB_machine_fetchstore8 (volatile void *ptr, __int64 value );
    void __TBB_machine_store8 (volatile void *ptr, __int64 value );
    __int64 __TBB_machine_load8 (const volatile void *ptr);
    bool __TBB_machine_trylockbyte ( volatile unsigned char& flag );
}

template <typename T, size_t S>
struct __TBB_machine_load_store {
    static inline T load_with_acquire(const volatile T& location) {
        T to_return = location;
#ifdef __TBB_fence_for_acquire 
        __TBB_fence_for_acquire();
#endif /* __TBB_fence_for_acquire */
        return to_return;
    }

    static inline void store_with_release(volatile T &location, T value) {
#ifdef __TBB_fence_for_release
        __TBB_fence_for_release();
#endif /* __TBB_fence_for_release */
        location = value;
    }
};

template <typename T>
struct __TBB_machine_load_store<T,8> {
    static inline T load_with_acquire(const volatile T& location) {
        return __TBB_machine_load8((volatile void *)&location);
    }

    static inline void store_with_release(T &location, T value) {
        __TBB_machine_store8((volatile void *)&location,(__int64)value);
    }
};

template<typename T>
inline T __TBB_machine_load_with_acquire(const volatile T &location) {
    return __TBB_machine_load_store<T,sizeof(T)>::load_with_acquire(location);
}

template<typename T, typename V>
inline void __TBB_machine_store_with_release(T& location, V value) {
    __TBB_machine_load_store<T,sizeof(T)>::store_with_release(location,value);
}

#define __TBB_load_with_acquire(L) __TBB_machine_load_with_acquire((L))
#define __TBB_store_with_release(L,V) __TBB_machine_store_with_release((L),(V))


#define DEFINE_ATOMICS(S,T,A,C) \
static inline T __TBB_machine_cmpswp##S ( volatile void * ptr, T value, T comparand ) { \
    T result; \
    volatile T *p = (T *)ptr; \
    __asm \
    { \
       __asm mov edx, p \
       __asm mov C , value \
       __asm mov A , comparand \
       __asm lock cmpxchg [edx], C \
       __asm mov result, A \
    } \
   __TBB_load_with_acquire(*(T *)ptr); \
    return result; \
} \
\
static inline T __TBB_machine_fetchadd##S ( volatile void * ptr, T addend ) { \
    T result; \
    volatile T *p = (T *)ptr; \
    __asm \
    { \
        __asm mov edx, p \
        __asm mov A, addend \
        __asm lock xadd [edx], A \
        __asm mov result, A \
    } \
   __TBB_load_with_acquire(*(T *)ptr); \
    return result; \
}\
\
static inline T __TBB_machine_fetchstore##S ( volatile void * ptr, T value ) { \
    T result; \
    volatile T *p = (T *)ptr; \
    __asm \
    { \
        __asm mov edx, p \
        __asm mov A, value \
        __asm lock xchg [edx], A \
        __asm mov result, A \
    } \
   __TBB_load_with_acquire(*(T *)ptr); \
    return result; \
}

DEFINE_ATOMICS(1, __int8, al, cl)
DEFINE_ATOMICS(2, __int16, ax, cx)
DEFINE_ATOMICS(4, __int32, eax, ecx)

static inline __int32 __TBB_machine_lg( unsigned __int64 i ) {
    unsigned __int32 j;
    __asm
    {
        bsr eax, i
        mov j, eax
    }
    return j;
}

static inline void __TBB_machine_OR( volatile void *operand, unsigned __int32 addend ) {
   __asm 
   {
       mov eax, addend
       mov edx, [operand]
       lock or [edx], eax
   }
}

static inline void __TBB_machine_AND( volatile void *operand, unsigned __int32 addend ) {
   __asm 
   {
       mov eax, addend
       mov edx, [operand]
       lock and [edx], eax
   }
}

static inline void __TBB_machine_pause (__int32 delay ) {
    _asm 
    {
        mov eax, delay
      L1: 
        pause
        add eax, -1
        jne L1  
    }
    return;
}

#define __TBB_CompareAndSwap1(P,V,C) __TBB_machine_cmpswp1(P,V,C)
#define __TBB_CompareAndSwap2(P,V,C) __TBB_machine_cmpswp2(P,V,C)
#define __TBB_CompareAndSwap4(P,V,C) __TBB_machine_cmpswp4(P,V,C)
#define __TBB_CompareAndSwap8(P,V,C) __TBB_machine_cmpswp8(P,V,C)
#define __TBB_CompareAndSwapW(P,V,C) __TBB_machine_cmpswp4(P,V,C)

#define __TBB_FetchAndAdd1(P,V) __TBB_machine_fetchadd1(P,V)
#define __TBB_FetchAndAdd2(P,V) __TBB_machine_fetchadd2(P,V)
#define __TBB_FetchAndAdd4(P,V) __TBB_machine_fetchadd4(P,V)
#define __TBB_FetchAndAdd8(P,V) __TBB_machine_fetchadd8(P,V)
#define __TBB_FetchAndAddW(P,V) __TBB_machine_fetchadd4(P,V)

#define __TBB_FetchAndStore1(P,V) __TBB_machine_fetchstore1(P,V)
#define __TBB_FetchAndStore2(P,V) __TBB_machine_fetchstore2(P,V)
#define __TBB_FetchAndStore4(P,V) __TBB_machine_fetchstore4(P,V)
#define __TBB_FetchAndStore8(P,V) __TBB_machine_fetchstore8(P,V)
#define __TBB_FetchAndStoreW(P,V) __TBB_machine_fetchstore4(P,V)

// Should define this: 
#define __TBB_Store8(P,V) __TBB_machine_store8(P,V)
#define __TBB_Load8(P) __TBB_machine_load8(P)
#define __TBB_AtomicOR(P,V) __TBB_machine_OR(P,V)
#define __TBB_AtomicAND(P,V) __TBB_machine_AND(P,V)

// Definition of other functions
#if !defined(_WIN32_WINNT)
extern "C" BOOL WINAPI SwitchToThread(void);
#endif
#define __TBB_Yield()  SwitchToThread()
#define __TBB_Pause(V) __TBB_machine_pause(V)
#define __TBB_Log2(V)    __TBB_machine_lg(V)

#define __TBB_TryLockByte(F) __TBB_machine_trylockbyte(F)

#define __TBB_cpuid
static inline void __TBB_x86_cpuid( __int32 buffer[4], __int32 mode ) {
    __asm
    {
        mov eax,mode
        cpuid
        mov edi,buffer
        mov [edi+0],eax
        mov [edi+4],ebx
        mov [edi+8],ecx
        mov [edi+12],edx
    }
}

#if defined(_MSC_VER) && defined(_Wp64)
    // Workaround for overzealous compiler warnings in /Wp64 mode
    #pragma warning (pop)
#endif /* _MSC_VER && _Wp64 */
