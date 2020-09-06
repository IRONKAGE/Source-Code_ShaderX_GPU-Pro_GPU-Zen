#ifndef _ProfilerConfig_h_
#define _ProfilerConfig_h_

// #define PROFILER_NO_CUDA_GL_TIMERS
#define PROFILER_SEND_IN_THREAD 1


#if defined(_WIN32)
#	if !defined(PROFILER_USE_TCP_SOCKETS)
#		define PROFILER_USE_WIN32_NAMED_PIPES 1
#	endif // PROFILER_USE_TCP_SOCKETS

/* XXX- Untested on windows. Let's not aggravate Ola. Disable for the moment
 * on windows. :-)
 */
#	undef PROFILER_SEND_IN_THREAD

#else
#	if !defined(PROFILER_USE_TCP_SOCKETS)
#		define PROFILER_USE_TCP_SOCKETS 1
#	endif // PROFILER_USE_TCP_SOCKETS

#endif // platform dependent

#endif // _ProfilerConfig_h_

