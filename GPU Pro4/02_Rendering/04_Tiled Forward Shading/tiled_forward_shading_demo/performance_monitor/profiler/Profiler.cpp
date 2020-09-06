#include "Profiler.h"

#include <cstring>

#include <utils/Assert.h>
#include <utils/IntTypes.h>

#if defined(PROFILER_USE_WIN32_NAMED_PIPES)
#	include <windows.h>

#elif defined(PROFILER_USE_TCP_SOCKETS)
#	if defined(_WIN32)
#		include <windows.h>
//#		include <winsock2.h>
typedef int ssize_t;
#		define TCPSEND_DEFAULT_FLAGS 0
#	else // other platforms
#		include <fcntl.h>
#		include <errno.h>
#		include <netdb.h>

#		include <sys/types.h>
#		include <sys/socket.h>
#		include <netinet/in.h>

#		define TCPSEND_DEFAULT_FLAGS MSG_NOSIGNAL
#	endif // platform 

#endif // PROFILER_USE_*

#ifndef PROFILER_NO_CUDA_GL_TIMERS

#include <GL/glew.h>

#ifndef DISABLE_CUDA
#include <cuda_runtime_api.h>
#endif // DISABLE_CUDA
#endif // PROFILER_NO_CUDA_GL_TIMERS

#if defined(PROFILER_SEND_IN_THREAD)
#	include <boost/thread/locks.hpp>
#	include <boost/thread/mutex.hpp>
#	include <boost/thread/thread.hpp>
#	include <boost/thread/condition_variable.hpp>
#endif // PROFILER_SEND_IN_THREAD

#ifdef GL_ARB_timer_query
  #define ENABLE_ASYNC_GL_TIMER 0
#else // GL_ARB_timer_query
  #define ENABLE_ASYNC_GL_TIMER 0
#endif // GL_ARB_timer_query

#define ENABLE_ASYNC_CUDA_TIMER 0

#ifndef PROFILER_NO_CUDA_GL_TIMERS

#if defined(PROFILER_USE_TCP_SOCKETS)
const char* kDefaultHostName = "localhost";
const char* kDefaultHostPort = "49374";

// blocking connect to server. returns socket.
static int connect_to_server( const char* addr, const char* port );

// close socket (close() or closesocket() depending on platform)
static void socket_close( int sock );
// set nonblocking mode on socket.
static bool socket_set_nonblock( int sock );


struct Profiler::ThreadData
{
	boost::mutex mutex;
	boost::thread thread;
	boost::condition_variable condition;

	template< typename F, typename T > ThreadData( F f, T t )
		: thread( f, t )
	{}
};
#endif // PROFILER_USE_TCP_SOCKETS

class GLTimer
{
public:
  enum { s_queryPoolSize = 1024 };

  void start()
  {
    m_timer = 0.0;
#if ENABLE_ASYNC_GL_TIMER
    m_glTimestampQueryPool.resize(s_queryPoolSize);
    glGenQueries(s_queryPoolSize, &m_glTimestampQueryPool[0]);
#endif // ENABLE_ASYNC_GL_TIMER
    // used for synchronous query
    glGenQueries(1, &m_glTimerQuery);
	  glBeginQuery(GL_TIME_ELAPSED_EXT, m_glTimerQuery);
  }

  //void stop();

  double getElapsedTime()
  {
	  glEndQuery(GL_TIME_ELAPSED_EXT);
    GLuint64 timeElapsed;
    // note, this syncs, which could be avoided using a pool of queries,
    // though then the idea of a time stamp need to be revised.
	  glGetQueryObjectui64vEXT(m_glTimerQuery, GL_QUERY_RESULT, &timeElapsed);
    // accumulate time elapsed thus far
    m_timer += double(timeElapsed) / 1e9;

    // restart timer so we can ask again...
	  glBeginQuery(GL_TIME_ELAPSED_EXT, m_glTimerQuery);
    
    return m_timer;
  }

#if ENABLE_ASYNC_GL_TIMER
  struct AsyncTimer
  {
    GLuint query;
    size_t timeCommandBufferOffset;
  };

  bool isAsyncAvaliable()
  {
    return GLEW_ARB_timer_query != 0;
  }

  void getElapsedTimeAsync(size_t cmdBufferOffset)
  {
    ASSERT(isAsyncAvaliable());
    // allocate more queries if we run out (may cause a hickup, but is perhaps better than crashing)...
    if (m_glTimestampQueryPool.empty())
    {
      m_glTimestampQueryPool.resize(s_queryPoolSize);
      glGenQueries(s_queryPoolSize, &m_glTimestampQueryPool[0]);
    }
    GLuint q = m_glTimestampQueryPool[m_glTimestampQueryPool.size() - 1];
    glQueryCounter(q, GL_TIMESTAMP);
    m_glTimestampQueryPool.pop_back();

    AsyncTimer at = { q, cmdBufferOffset };
    m_asyncTimers.push_back(at);
  }


  // call to wait for all async timers to complete and write their result to the destination.
  void flushAsyncTimers(uint8_t *cmdBuffer)
  {
    if (isAsyncAvaliable())
    {
      // get sync time to force sync, i.e all async should be ready after
      getElapsedTime();

      // async should be ready now...
      for (size_t i = 0; i < m_asyncTimers.size(); ++i)
      {
        AsyncTimer &at = m_asyncTimers[i];

        GLuint64 timeStamp = 0;
        glGetQueryObjectui64v(at.query, GL_QUERY_RESULT, &timeStamp);

        *reinterpret_cast<double*>(cmdBuffer + at.timeCommandBufferOffset) = double(timeStamp) / 1e9;
        
        // event is now free for re-use
        m_glTimestampQueryPool.push_back(at.query);
      }
      m_asyncTimers.clear();
    }
  }
#endif // ENABLE_ASYNC_GL_TIMER

protected:
  double m_timer;
#ifndef PROFILER_NO_CUDA_GL_TIMERS
  GLuint m_glTimerQuery;
#if ENABLE_ASYNC_GL_TIMER
  std::vector<AsyncTimer> m_asyncTimers;
  std::vector<GLuint> m_glTimestampQueryPool;
#endif // ENABLE_ASYNC_GL_TIMER

#endif // PROFILER_NO_CUDA_GL_TIMERS
};



class CudaTimer
{
public:
#ifndef DISABLE_CUDA
  enum { s_eventPoolSize = 1024 };

  void start()
  {
    //m_timer = 0.0;
    m_startEvent = 0;
    cudaEventCreate(&m_startEvent);
    cudaEventCreate(&m_endEvent);
    cudaEventRecord(m_startEvent, 0);
    m_eventPool.resize(s_eventPoolSize);
    // fill pool of free events
    for (size_t i = 0; i < m_eventPool.size(); ++i)
    {
      cudaEventCreate(&m_eventPool[i]);
    }
  }

  double getElapsedTime()
  {
    cudaEventRecord(m_endEvent, 0);
    cudaEventSynchronize(m_endEvent);
    float dt = 0.0f;
    cudaEventElapsedTime(&dt, m_startEvent, m_endEvent);
    //cudaEventRecord(m_startEvent, 0);

    //m_timer += dt / 1000.0;
    return dt / 1000.0;//m_timer;
  }

  struct AsyncTimer
  {
    cudaEvent_t event;
    size_t timeCommandBufferOffset;
  };

  void getElapsedTimeAsync(size_t cmdBufferOffset)
  {
    if(m_eventPool.empty())
    {
      m_eventPool.resize(s_eventPoolSize);
      // fill pool of free events
      for (size_t i = 0; i < m_eventPool.size(); ++i)
      {
        cudaEventCreate(&m_eventPool[i]);
      }
    }
    cudaEvent_t e = m_eventPool[m_eventPool.size() - 1];
    cudaEventRecord(e, 0);
    m_eventPool.pop_back();

    AsyncTimer at = { e, cmdBufferOffset };
    m_asyncTimers.push_back(at);
  }


  // call to wait for all async timers to complete and write their result to the command buffer.
  void flushAsyncTimers(uint8_t *cmdBuffer)
  {
    // we record and sync this event to force all others to be completed also
    cudaEventRecord(m_endEvent, 0);
    cudaEventSynchronize(m_endEvent);

    // they must now be recorded
    for (size_t i = 0; i < m_asyncTimers.size(); ++i)
    {
      AsyncTimer &at = m_asyncTimers[i];

      float dt = 0.0f;
      cudaEventElapsedTime(&dt, m_startEvent, at.event);
      *reinterpret_cast<double*>(cmdBuffer + at.timeCommandBufferOffset) = dt / 1000.0;
      
      // event is now free for re-use
      m_eventPool.push_back(at.event);
    }
    m_asyncTimers.clear();
  }

  std::vector<cudaEvent_t> m_eventPool;
  std::vector<AsyncTimer> m_asyncTimers;
  cudaEvent_t m_startEvent;
  cudaEvent_t m_endEvent; // used for syncronized timers, and for flushing the async timers

#endif // DISABLE_CUDA
};

#endif // PROFILER_NO_CUDA_GL_TIMERS










void Profiler::reset()
{
  PerformanceTimer m_timer;

  m_stackOffset = 0;
  m_tokenTable.clear();
  m_globalCounters.clear();
  m_commandBuffer.clear();  
}

Profiler &Profiler::instance()
{
  static Profiler inst;
  return inst;
}



uint32_t Profiler::registerToken(const char *token)
{
  TokenTable::iterator it = m_tokenTable.find(token);
  if (it != m_tokenTable.end())
  {
    return (*it).object;
  }

  uint32_t id = uint32_t(m_tokenTable.size());
  m_tokenTable.insert(token, id);
  m_globalCounters.push_back(-1.0);

  size_t strSize = strlen(token) + 1;
  
  Command *cmd = createCommand(CID_NewToken, strSize, TT_Cpu);
  cmd->m_sizeOfNewToken = uint32_t(strSize);
  memcpy(cmd + 1, token, strSize);

  return id;
}



void Profiler::beginBlock(uint32_t tokenId, uint32_t timerType)
{
  Command *cmd = createCommand(CID_BeginBlock, 0, timerType);
  cmd->m_tokenId = tokenId;
  
  pushToken(tokenId, timerType);
}



void Profiler::endBlock()
{
  StackItem t = popToken();
  Command *cmd = createCommand(CID_EndBlock, 0, t.timerType);
  cmd->m_tokenId = t.tokenId;


  // if the stack is 0 here, we may as well flush counters and async timers?
  if (m_stackOffset == 0)
  {
    flushGlobals();
#ifndef PROFILER_NO_CUDA_GL_TIMERS
#ifndef DISABLE_CUDA
    if (m_cudaTimer)
    {
      m_cudaTimer->flushAsyncTimers(&m_commandBuffer[0]);
    }
#endif // DISABLE_CUDA
#if ENABLE_ASYNC_GL_TIMER
    if (m_glTimer)
    {
      m_glTimer->flushAsyncTimers(&m_commandBuffer[0]);
    }
#endif // ENABLE_ASYNC_GL_TIMER
#endif // PROFILER_NO_CUDA_GL_TIMERS
  }
}



void Profiler::addCounter(uint32_t token, double value, NameSpace ns)
{
  if (ns == NS_Local)
  {
    Command *cmd = createCommand(CID_Counter, 0, TT_Cpu);
    cmd->m_tokenId = token;
    cmd->m_value = value;
  }
  else
  {
    double &v = m_globalCounters[token];
    v = (v < 0.0) ? value : (v + value);
  }
}



void Profiler::addMessage(uint32_t token, const char *message)
{
  size_t strSize = strlen(message) + 1;
  
  Command *cmd = createCommand(CID_Message, strSize, TT_Cpu);
  cmd->m_sizeOfMessage = uint32_t(strSize);
  cmd->m_tokenId = token;
  memcpy(cmd + 1, message, strSize);
}



void Profiler::addEvent(uint32_t token, uint32_t timerType)
{
  Command *cmd = createCommand(CID_OneOff, 0, timerType);
  cmd->m_tokenId = token;
}



const char *Profiler::getTokenString(uint32_t tokenId) const
{
  ASSERT(tokenId < m_tokenTable.size());
  return ((m_tokenTable.begin() + tokenId)->key).c_str();
}



void Profiler::clearCommandBuffer()
{
  // check that we are not in a begin/end block.
  ASSERT(m_stackOffset == 0); 

  m_commandBuffer.clear();
}



void Profiler::setCommandBuffer(const uint8_t *buf, const size_t size)
{
  // check that we are not in a begin/end block.
  ASSERT(m_stackOffset == 0); 

  m_commandBuffer.clear();
  m_commandBuffer.insert(m_commandBuffer.begin(), buf, buf + size);
  for (const_iterator it = begin(); it != end(); ++it)
  {
    const Command &cmd = (*it);
    if (cmd.m_id == CID_NewToken)
    {
      const char *token = reinterpret_cast<const char *>(cmd.extraData());

      uint32_t id = uint32_t(m_tokenTable.size());
      m_tokenTable.insert(token, id);

    }
  }
}



static bool spawnProcess(const std::string &progname)
{
#if defined(_WIN32)
	STARTUPINFO sInfo;
	memset(&sInfo, 0, sizeof(sInfo));
	sInfo.cb = sizeof(sInfo);

	PROCESS_INFORMATION processInfo;
	memset(&processInfo, 0, sizeof(processInfo));

	if(0 == CreateProcess(progname.c_str(), NULL, NULL, NULL, FALSE, DETACHED_PROCESS | NORMAL_PRIORITY_CLASS, NULL, NULL, &sInfo, &processInfo))
	{
		return false;
		//printf("Failed to create process, error: %d\n", GetLastError());
	}

	return true;
#elif defined(__linux__)
	// TODO
	return false;
#else
	return false;
#endif // platform dependent
}



bool Profiler::launchProfileMonitor(const char *binPath, bool connect)
{
#if defined(_WIN32)
	if (spawnProcess(std::string(binPath) + "\\performance_monitor_net2.exe"))
	{
		if (connect)
		{
			return connectToPerformanceMonitor();
		}
		return true;
	}
	return false;
#elif defined(__linux__)
	return connectToPerformanceMonitor();
#else
	return false;
#endif // platform dependent
}



bool Profiler::connectToPerformanceMonitor()
{
#if defined(PROFILER_SEND_IN_THREAD)
	//m_sendThread = boost::thread( &Profiler::sendThreadFunc, this );
	m_threadData = new ThreadData( &Profiler::sendThreadFunc, this );
#endif // PROFILER_SEND_IN_THREAD

#if defined(PROFILER_USE_WIN32_NAMED_PIPES)
  LPTSTR lpszPipename = TEXT("\\\\.\\pipe\\performance_monitor");

  for (int i = 0; i < 200 && m_performancePipeHandle == INVALID_HANDLE_VALUE; ++i)
  {
    m_performancePipeHandle = CreateFile( 
      lpszPipename,   // pipe name 
      GENERIC_WRITE, 
      0,              // no sharing 
      NULL,           // default security attributes
      OPEN_EXISTING,  // opens existing pipe 
      0,              // default attributes 
      NULL);          // no template file 

		if (m_performancePipeHandle == INVALID_HANDLE_VALUE)
		{
			Sleep(10);
		}
	}

  if (m_performancePipeHandle != INVALID_HANDLE_VALUE) 
  {
    DWORD dwMode = PIPE_READMODE_MESSAGE; 
    BOOL fSuccess = SetNamedPipeHandleState( 
      m_performancePipeHandle,    // pipe handle 
      &dwMode,  // new pipe mode 
      NULL,     // don't set maximum bytes 
      NULL);    // don't set maximum time 
    if (!fSuccess) 
    {
      printf("SetNamedPipeHandleState failed\n"); 
      CloseHandle(m_performancePipeHandle);
      m_performancePipeHandle = INVALID_HANDLE_VALUE;
      return false;
    }
  }
  else
  {
    if (GetLastError() != ERROR_PIPE_BUSY) 
    {
      printf("Could not open pipe\n"); 
    }
    return false;
  }
  return true;

#elif defined(PROFILER_USE_TCP_SOCKETS)
	int sock = connect_to_server( kDefaultHostName, kDefaultHostPort );
	if( -1 == sock ) 
	{
		fprintf( stderr, "connect_to_server failed. (host=%s, port=%s)\n",
			kDefaultHostName, kDefaultHostPort 
		);
		return false;
	}
	
	m_socketHandle = sock;
	return true;
#else // no connections!
	return false;
#endif // connection kind
}



bool Profiler::sendResultsToMonitor( bool clearBuffer )
{
#if !defined(PROFILER_SEND_IN_THREAD)
	bool ret = sendResultsToMonitorImpl_( m_commandBuffer );

	if( clearBuffer )
		clearCommandBuffer();
	
	return ret;
#else // SEND_IN_THREAD
	/* Note/Warning: there's currently no guarantee that the thread sending
	 * profiling data will send *all* command buffers. This depends on the
	 * relative speed between the main thread issuing sendResultsToMonitor()
	 * and the worker thread. 
	 *
	 * Current behaviour is that the worker thread will drop command buffers
	 * if it's too slow. This may cause missing token-updates (bad).
	 */
	ASSERT( m_threadData );

	{
		boost::unique_lock<boost::mutex> lock( m_threadData->mutex );
		std::swap( m_commandBuffer, m_transferBuffer );
	}
	m_threadData->condition.notify_one();

	if( clearBuffer )
		clearCommandBuffer();

	return true;
#endif // PROFILER_SEND_IN_THREAD
}

bool Profiler::sendResultsToMonitorImpl_( const std::vector<uint8_t>& sendBuffer )
{
#if defined(PROFILER_USE_WIN32_NAMED_PIPES)
	BOOL ok = TRUE;

	if ( m_performancePipeHandle != INVALID_HANDLE_VALUE && !sendBuffer.empty() ) 
	{
		DWORD cbWritten = 0;
		ok = WriteFile(m_performancePipeHandle, &sendBuffer[0], uint32_t(sendBuffer.size()), &cbWritten, 0);

		if (!ok) 
		{
			printf("printPerformanceResults: Pipe write failed (%llu bytes), closing.\n", m_commandBuffer.size());
			CloseHandle(m_performancePipeHandle);
			m_performancePipeHandle = INVALID_HANDLE_VALUE;
		}
	}

	return ok == TRUE;

#elif defined(PROFILER_USE_TCP_SOCKETS)
	if( -1 != m_socketHandle  && !sendBuffer.empty() )
	{
		size_t sentData = 0;
		uint64_t sendBufferSize = sendBuffer.size();

		const uint8_t* sendBufferPtr = reinterpret_cast<uint8_t*>(&sendBufferSize);
		for( sentData = 0; sizeof(uint64_t) != sentData; )
		{
			ssize_t ret = send( m_socketHandle, sendBufferPtr+sentData, 
				sizeof(uint64_t)-sentData, TCPSEND_DEFAULT_FLAGS );

			if( -1 == ret )
			{
				fprintf( stderr, "send(sendSize) - error: `%s'\n", strerror(errno) );
				fprintf( stderr, "  shutting down socket. no further comm!\n" );

				socket_close( m_socketHandle );
				m_socketHandle = -1;

				return false;
			}

			sentData += ret;
		}

		sendBufferPtr = reinterpret_cast<const uint8_t*>(&sendBuffer[0]);
		for( sentData = 0; sendBufferSize != sentData; )
		{
			ssize_t ret = send( m_socketHandle, sendBufferPtr+sentData,
				sendBufferSize-sentData, TCPSEND_DEFAULT_FLAGS );

			if( -1 == ret )
			{
				fprintf( stderr, "send(sendBuffer) - error: `%s'\n", strerror(errno) );
				fprintf( stderr, "  shutting down socket. no further comm!\n" );

				socket_close( m_socketHandle );
				m_socketHandle = -1;

				return false;
			}

			sentData += ret;
		}
	}

	return true;

#else // no connection
	return false;
#endif // PROFILER_USE_*
}

void Profiler::disconnectFromMonitor()
{
#if defined(PROFILER_USE_WIN32_NAMED_PIPES)
	CloseHandle(m_performancePipeHandle); 
	m_performancePipeHandle = INVALID_HANDLE_VALUE;
#elif defined(PROFILER_USE_TCP_SOCKETS)
	socket_close(m_socketHandle);
	m_socketHandle = -1;
#endif // PROFILER_USE_*
}


void Profiler::flushGlobals()
{
  // flush global counters.
  for (uint32_t i = 0; i < uint32_t(m_globalCounters.size()); ++i)
  {
    double v = m_globalCounters[i];
    if (v > 0.0)
    {
      addCounter(i, v);
    }
    m_globalCounters[i] = -1.0;
  }
}


Profiler::Command *Profiler::createCommand(CommandId id, size_t extraBytes, uint32_t timerType)
{
  size_t offset = m_commandBuffer.size();
  size_t timeStampMemberOffset = (size_t)(&((Command*)0)->m_timeStamp);
  // default is CPU anyway...
  double timeStamp = m_timer.getElapsedTime();
#ifndef PROFILER_NO_CUDA_GL_TIMERS
  switch(timerType) 
  {
  case TT_OpenGl:
    if (m_glTimer)
    {
#if ENABLE_ASYNC_GL_TIMER
      if (m_glTimer->isAsyncAvaliable())
      {
        m_glTimer->getElapsedTimeAsync(offset + timeStampMemberOffset);
        timeStamp = 0.0;
      }
      else
#endif // ENABLE_ASYNC_GL_TIMER
      {
        timeStamp = m_glTimer->getElapsedTime();
      }
    }
    break;
#ifndef DISABLE_CUDA 
  case TT_Cuda:
    {
    //timeStamp = m_cudaTimer->getElapsedTime();
      if (m_cudaTimer)
      {
#if ENABLE_ASYNC_CUDA_TIMER
        m_cudaTimer->getElapsedTimeAsync(offset + timeStampMemberOffset);
        timeStamp = 0.0;
#else // ENABLE_ASYNC_CUDA_TIMER
        timeStamp = m_cudaTimer->getElapsedTime();
#endif // ENABLE_ASYNC_CUDA_TIMER
      }
    }
    break;
#endif // DISABLE_CUDA
  default:
    break;
  };
#endif // PROFILER_NO_CUDA_GL_TIMERS
  m_commandBuffer.resize(offset + sizeof(Command) + extraBytes);
  Command *cmd = reinterpret_cast<Command*>(&m_commandBuffer[offset]);
  cmd->m_id = id;
  cmd->m_timeStamp = timeStamp;
  return cmd;
}


uint32_t Profiler::s_enabledTimers = Profiler::TT_All;

void Profiler::setEnabledTimers(uint32_t timers)
{
  s_enabledTimers = timers;
}

Profiler::Profiler() :
  m_tokenTable(16 * 1024),
  m_stackOffset(0)
{
  // start cpu timer
  m_timer.start();
  m_glTimer = 0;
  m_cudaTimer = 0;
#ifndef PROFILER_NO_CUDA_GL_TIMERS
  if (s_enabledTimers & TT_OpenGl)
  {
    // Start opengl timer
    m_glTimer = new GLTimer;
    m_glTimer->start();
  }
  //
#ifndef DISABLE_CUDA
	if (s_enabledTimers & TT_Cuda)
  {
    m_cudaTimer = new CudaTimer;
    m_cudaTimer->start();
  }
#endif // DISABLE_CUDA
#endif // PROFILER_NO_CUDA_GL_TIMERS

#if defined(PROFILER_USE_WIN32_NAMED_PIPES)
	m_performancePipeHandle = INVALID_HANDLE_VALUE;
#elif defined(PROFILER_USE_TCP_SOCKETS)
	m_socketHandle = -1;
#endif // connection kind
}

// -- SEND_IN_THREAD stuff --
#if defined(PROFILER_SEND_IN_THREAD)
void Profiler::sendThreadFunc()
{
	ASSERT( m_threadData );
	std::vector<uint8_t> sendBuffer;

	for( ;; )
	{
		// get data from other thread
		{
			boost::unique_lock<boost::mutex> lock( m_threadData->mutex );
			m_threadData->condition.wait( lock );
		
			std::swap( sendBuffer, m_transferBuffer );
		}

		// send data
		if( !sendResultsToMonitorImpl_( sendBuffer ) )
		{
			fprintf( stderr, "error: unable to send results to monitor.\n" );
			fprintf( stderr, "   terminating sending thread.\n" );
			break;
		}
	}
}
#endif // PROFILER_SEND_IN_THREAD

// -- USE_TCP_SOCKETS stuff --
#if defined(PROFILER_USE_TCP_SOCKETS)
static int connect_to_server( const char* addr, const char* port )
{
	sockaddr_in servAddr;
	memset( &servAddr, 0, sizeof(servAddr) );

	// resolve server. uses getaddrinfo(), which should also be available on
	// windows.
	{
		addrinfo hints;
		memset( &hints, 0, sizeof(hints) );
		hints.ai_family = AF_INET;
		hints.ai_socktype = SOCK_STREAM;
		hints.ai_protocol = IPPROTO_TCP;

		addrinfo* result = 0;
		int ret = getaddrinfo( addr, port, &hints, &result );
		
		if( 0 != ret )
		{
			fprintf( stderr, "Error - cannot resolve address: %s\n",
				gai_strerror(ret) 
			);

			return -1;
		}
		
		bool ok = false;
		for( addrinfo* res = result; res; res = res->ai_next )
		{
			if( res->ai_family == AF_INET 
				&& res->ai_addrlen == sizeof(sockaddr_in) )
			{
				ok = true;
				memcpy( &servAddr, res->ai_addr, sizeof(sockaddr_in) );
				break;
			}
		}

		freeaddrinfo( result );

		if( !ok )
		{
			fprintf( stderr, "Error - no appropriate address format\n" );
			return -1;
		}
	}

	// allocate socket
	int fd = socket( AF_INET, SOCK_STREAM, 0 );
	
	if( -1 == fd )
	{
		perror( "socket() failed" );
		return -1;
	}

	// attempt to establish connection
	if( -1 == connect( fd, (const sockaddr*)&servAddr, sizeof(servAddr) ) )
	{
		perror( "connect() failed" );
		socket_close( fd );
		return -1;
	}

	// ok
	return fd;
}

static void socket_close( int fd )
{
#	if defined(_WIN32)
	closesocket( fd );
#	else
	close( fd );
#	endif // platform dep
}
static bool socket_set_nonblock( int fd )
{
#	if defined(_WIN32)
	long arg = 1;

	if( SOCKET_ERROR == ioctlsocket( fd, FIONBIO, &arg ) )
	{
		// TODO - someting with FormatMessage().
		int err = WSAGetLastError();
		fprintf( stderr, "ioctlsocket(FIONBIO) failed: %d\n", err );
		return false;
	}
#	else
	int oldFlags = fcntl( fd, F_GETFL, 0 );
	if( -1 == oldFlags )
	{
		perror( "fcntl(F_GETFL) failed" );
		return false;
	}

	if( -1 == fcntl( fd, F_SETFL, oldFlags | O_NONBLOCK ) )
	{
		perror( "fcntl(F_SETFL) failed" );
		return false;
	}
#	endif // platform dep

	return true;
}

#endif // PROFILER_USE_TCP_SOCKETS
// !USE_TCP_SOCKETS stuff
