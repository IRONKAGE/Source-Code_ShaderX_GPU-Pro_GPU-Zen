#ifndef _Profiler_h_
#define _Profiler_h_

#include "ProfilerConfig.h"

#include <vector>

#include <utils/HashVector.h>
#include <utils/PlatformCompat.h>
#include <utils/PerformanceTimer.h>

class GLTimer;
class CudaTimer;

/**
 * The profiler records a stream of events, which can be packaged off to disc of interpreted at runtime,
 * possibly these can also be shipped off to another process for real time graphical display.
 * The events are simple binary commands, of varying length, that are used to recreate timings. The benefit
 * of using events are that we can ignore those we are not interested in, and also re-interpret the data later.
 * An event can be a new token, a message, a counter, a one off event, or a begin/end bracket.
 *  EVENT ::= new_token_cmd size_of_token char[size_of_token]
 *        |   message_cmd size_of_message char[size_of_message]
 *        |   counter_cmd token_id value : int time_value : double
 *        |   one_off token_id time_value : double
 *        |   begin token_id time_value : double
 *        |   end token_id time_value : double
 * From these we can reconstruct complicated relations such as nested timers and multiple frames, for example:
 *  new token "frame" => 0
 *  begin 0 0.01
 *   new token "timer A" => 1
 *   begin 1 0.011
 *   end 1 0.02
 *  end 0 0.021
 *  new token "TraversalMethodB" => 2
 *  one off event 2, 0.022
 *  ...
 * The system takes care of allocating the tokens as necessary and inserting these commands into the stream, the stream 
 * thus contains the information needed to display it. When decoding it the user needs to rebuild the mapping from id to
 * string, or get hold of the original table. Note that in the inter-process scenario we may be forced to send the table when
 * the monitoring application comes online. Or we may need to insert periodical key-frames which records all tokens.
 */
class Profiler
{
public:
  typedef HashVector<std::string, uint32_t> TokenTable;

  Profiler();

  enum TimerType
  {
    TT_Cpu = 1,
    TT_Cuda = 1 << 1,
    TT_OpenGl = 1 << 2,
    TT_All = TT_OpenGl | TT_Cuda | TT_Cpu
  };

  /**
   * This may be called BEFORE any call to instance() (at which point the singleton is creater) to set which timers should exist.
   * It does not create the singleton, and can thus be called safely before initializing cuda or open gl.
   * for example: Profiler::setEnabledTimers(TT_Cpu | TT_OpenGl);
   */
  static void setEnabledTimers(uint32_t timers);

  enum NameSpace
  {
    NS_Local,
    NS_Global,
    NS_Max
  };


  enum CommandId
  {
    CID_NewToken,
    CID_Message,
    CID_Counter,
    CID_BeginBlock,
    CID_EndBlock,
    CID_OneOff,
    CID_Max,
  };


	struct Command
  {
    double m_timeStamp;
    uint32_t m_id;
		
		BEGIN_WARNING_CLOBBER_MSVC
    union
    {
      uint32_t m_sizeOfNewToken;
      struct // for messages
      {
        uint32_t m_tokenId;
        uint32_t m_sizeOfMessage;
      };
      struct // for counters
      {
        uint32_t m_tokenId_; //XXX-WARNING same as m_tokenId above
        double m_value;
      };
    };
		END_WARNING_CLOBBER_MSVC

    const uint8_t* extraData() const { return reinterpret_cast<const uint8_t*>(this + 1); }
    uint8_t* extraData() { return reinterpret_cast<uint8_t*>(this + 1); }
  };
	

  /**
   */
  void reset();

  /**
   */
  static Profiler &instance();

  /**
   * Will keep a pointer to the token string, it must thus be kept alive by the user, the usual
   * usage pattern is to use a string literal and in this case it is not a problem. The returned
   * number is a handle that can be used in subsequent calls to identify the token. 
   * If the token has not been seen before, then a new token command is generated.
   */
  uint32_t registerToken(const char *token);

  /**
   */
  void beginBlock(uint32_t tokenId, uint32_t timerType = TT_Cpu);
  
  /**
   */
  void endBlock();

  /**
   */
  void addCounter(uint32_t token, double value, NameSpace ns = NS_Local);

  /**
   */
  void addMessage(uint32_t token, const char *message);
	void addMessage(uint32_t token, const std::string &message) { addMessage(token, message.c_str()); }

  /**
   */
  void addEvent(uint32_t token, uint32_t timerType = TT_Cpu);
  

  class const_iterator
  {
  public:
    const_iterator(const std::vector<uint8_t>::const_iterator &it) : m_it(it) { }

    const Command& operator *() const { return *reinterpret_cast<const Command *>(&*m_it); }
	const Command* operator ->() const { return reinterpret_cast<const Command*>(&*m_it); }

    bool operator != (const const_iterator &o) { return m_it != o.m_it; }
    void operator++()
    {
      const Command& cmd = **this;
      const size_t skip = sizeof(Command) + ((cmd.m_id == CID_Message) ? cmd.m_sizeOfMessage : ((cmd.m_id == CID_NewToken) ? cmd.m_sizeOfNewToken : 0));
      m_it += skip;
    }

  protected:
    std::vector<uint8_t>::const_iterator m_it;
  };

  const_iterator begin() const { return const_iterator(m_commandBuffer.begin()); }
  const_iterator end() const { return const_iterator(m_commandBuffer.end()); }

  /**
   */
  const char *getTokenString(uint32_t tokenId) const;

  /**
   */
  void clearCommandBuffer();

  const std::vector<uint8_t> &getCommandBuffer() { return m_commandBuffer; }
  void setCommandBuffer(const std::vector<uint8_t> &buf) { m_commandBuffer = buf; }
  void setCommandBuffer(const uint8_t *buf, const size_t size);

  typedef TokenTable::iterator token_iterator;
  token_iterator beginTokens()  { return m_tokenTable.begin(); }
  token_iterator endTokens()    { return m_tokenTable.end(); }

  size_t tokenCount() const { return m_tokenTable.size(); }

  /**
   */
  bool launchProfileMonitor(const char *binPath = "", bool connect = true);
  /**
   */
  bool connectToPerformanceMonitor();
  /**
   */
  bool sendResultsToMonitor(bool clearCommandBuffer);
  /**
   */
  void disconnectFromMonitor();

private:
  /**
   * Generates commands for all global counters and timers, also resets them to 0.
   * Is automatically called each time the token stack is down to 0.
   */
  void flushGlobals();

  Command *createCommand(CommandId id, size_t extraBytes, uint32_t timerType);

  struct StackItem
  {
    uint32_t tokenId;
    uint32_t timerType;
  };

  void pushToken(uint32_t tokenId, uint32_t timerType)
  {
    ASSERT(m_stackOffset < s_maxStack);
    m_tokenStack[m_stackOffset].timerType = timerType;
    m_tokenStack[m_stackOffset++].tokenId = tokenId;
  }

  StackItem popToken()
  {
    ASSERT(m_stackOffset > 0);
    return m_tokenStack[--m_stackOffset];
  }

  // not copyable
  void operator=(const Profiler&);
  Profiler(const Profiler&);
  // and singleton


  PerformanceTimer m_timer;
  GLTimer *m_glTimer;
  CudaTimer *m_cudaTimer;
  enum { s_maxStack = 100 * 1024 };
  StackItem m_tokenStack[s_maxStack];
  uint32_t m_stackOffset;


  TokenTable m_tokenTable;
  std::vector<double> m_globalCounters;
  std::vector<uint8_t> m_commandBuffer;

#	if defined(PROFILER_USE_WIN32_NAMED_PIPES)
	void *m_performancePipeHandle;
#	elif defined(PROFILER_USE_TCP_SOCKETS)
	int m_socketHandle;
#	endif // PROFILER_USE_*

	bool sendResultsToMonitorImpl_( 
		const std::vector<uint8_t>& results
	);

#	if defined(PROFILER_SEND_IN_THREAD)
	struct ThreadData;
	ThreadData* m_threadData;

	void sendThreadFunc();

	std::vector<uint8_t> m_transferBuffer;
#	endif

	public:
		static uint32_t s_enabledTimers;
};



struct ScopeBlockProfileHelper
{
  ScopeBlockProfileHelper(uint32_t token, uint32_t timerType = Profiler::TT_Cpu)
  {
    Profiler::instance().beginBlock(token, timerType);
  }



  ~ScopeBlockProfileHelper()
  {
    Profiler::instance().endBlock();
  }
};



// These macros are for instrumenting the code, they can then be compiled out when instrumentation is not desired.

#define CAT_ID2(_id1_, _id2_) _id1_##_id2_
#define CAT_ID(_id1_, _id2_) CAT_ID2(_id1_, _id2_)

#define PROFILE_SCOPE(_token_string_id_) \
  static uint32_t CAT_ID(_scope_profiler_token_, __LINE__) = Profiler::instance().registerToken(_token_string_id_); \
  ScopeBlockProfileHelper CAT_ID(_scope_helper, __LINE__)(CAT_ID(_scope_profiler_token_, __LINE__))

#define PROFILE_SCOPE_2(_token_string_id_, _timer_type_) \
  static uint32_t CAT_ID(_scope_profiler_token_, __LINE__) = Profiler::instance().registerToken(_token_string_id_); \
  ScopeBlockProfileHelper CAT_ID(_scope_helper, __LINE__)(CAT_ID(_scope_profiler_token_, __LINE__), Profiler::_timer_type_)

#define PROFILE_BEGIN_BLOCK(_token_string_id_) \
  static uint32_t CAT_ID(_scope_profiler_token_, __LINE__) = Profiler::instance().registerToken(_token_string_id_); \
  Profiler::instance().beginBlock(CAT_ID(_scope_profiler_token_, __LINE__))

#define PROFILE_BEGIN_BLOCK_2(_token_string_id_, _timer_type_) \
  static uint32_t CAT_ID(_scope_profiler_token_, __LINE__) = Profiler::instance().registerToken(_token_string_id_); \
  Profiler::instance().beginBlock(CAT_ID(_scope_profiler_token_, __LINE__), Profiler::_timer_type_)

#define PROFILE_END_BLOCK() \
  Profiler::instance().endBlock()

#define PROFILE_END_BLOCK_2() \
  Profiler::instance().endBlock()

#define PROFILE_MESSAGE(_token_string_id_, _message_string_) \
  static uint32_t CAT_ID(_scope_profiler_token_, __LINE__) = Profiler::instance().registerToken(_token_string_id_); \
  Profiler::instance().addMessage(CAT_ID(_scope_profiler_token_, __LINE__), _message_string_)

#define PROFILE_COUNTER(_token_string_id_, _value_) \
  static uint32_t CAT_ID(_scope_profiler_token_, __LINE__) = Profiler::instance().registerToken(_token_string_id_); \
  Profiler::instance().addCounter(CAT_ID(_scope_profiler_token_, __LINE__), _value_)

#define PROFILE_COUNTER_VAR(_token_string_id_, _value_) \
  Profiler::instance().addCounter(Profiler::instance().registerToken(_token_string_id_), _value_)

#define PROFILE_GLOBAL_COUNTER(_token_string_id_, _value_) \
  static uint32_t CAT_ID(_scope_profiler_token_, __LINE__) = Profiler::instance().registerToken(_token_string_id_); \
  Profiler::instance().addCounter(CAT_ID(_scope_profiler_token_, __LINE__), _value_, Profiler::NS_Global)

#define PROFILE_EVENT(_token_string_id_) \
  static uint32_t CAT_ID(_scope_profiler_token_, __LINE__) = Profiler::instance().registerToken(_token_string_id_); \
  Profiler::instance().addEvent(CAT_ID(_scope_profiler_token_, __LINE__))

#define PROFILE_EVENT_VAR(_token_string_id_) Profiler::instance().addEvent(Profiler::instance().registerToken(_token_string_id_))

#define PROFILE_SCOPE_CUDA_SYNC(_token_string_id_) \
  static uint32_t CAT_ID(_scope_profiler_token_, __LINE__) = Profiler::instance().registerToken(_token_string_id_); \
  ScopeBlockProfileHelper CAT_ID(_scope_helper, __LINE__)(CAT_ID(_scope_profiler_token_, __LINE__), Profiler::TT_Cuda)

#define PROFILE_BEGIN_BLOCK_CUDA_SYNC(_token_string_id_)   cudaThreadSynchronize(); PROFILE_BEGIN_BLOCK(_token_string_id_)
#define PROFILE_END_BLOCK_CUDA_SYNC()   cudaThreadSynchronize(); PROFILE_END_BLOCK()


#endif // _Profiler_h_
