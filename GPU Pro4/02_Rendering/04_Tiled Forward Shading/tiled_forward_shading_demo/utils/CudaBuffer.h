#ifndef _CudaBuffer_h_
#define _CudaBuffer_h_

#include <cstdio>
#include <cassert>
#include <iostream>

#include "Assert.h"
#include "CudaMath.h"
#include "CudaSafeCall.h"

template <typename T>
class CudaBuffer
{
public:
	typedef T *CudaPtr;

	CudaBuffer(uint32_t size = 0) : 
		m_size(0), 
		m_data(0),
		m_hostData(0),
		m_asyncInProgress(false)
	{ 
		cudaEventCreate(&m_asyncCopyEvent);
		init(size);
	} 

	CudaBuffer(const T *hostData, size_t size) : 
		m_size(0), 
		m_data(0),
		m_hostData(0),
		m_asyncInProgress(false)
	{ 
		cudaEventCreate(&m_asyncCopyEvent);
		init(uint32_t(size));
		copyFromHost(hostData, uint32_t(size));
	} 


	CudaBuffer(const CudaBuffer &other) : 
		m_size(0), 
		m_data(0),
		m_hostData(0),
		m_asyncInProgress(false)
	{
		cudaEventCreate(&m_asyncCopyEvent);

		init(other.size());
		copy(other, other.size());
	}
	~CudaBuffer()
	{
		cudaEventDestroy(m_asyncCopyEvent);
		clear();
	}

	void clear()
	{
		assert(!m_hostData);
		if (m_data)
		{
			CUDA_SAFE_CALL((cudaFree(m_data)));
		}
		m_size = 0; 
		m_data = 0;
	}

	void init(uint32_t size)
	{
		resize(size);
	}

	void resize(uint32_t size, bool growOnly = false, bool keepData = false)
	{
		if (!growOnly || size > m_size)
		{
			T *oldData = 0;
			uint32_t oldSize = 0;
			if (m_data)
			{
				// defer freeing of data...
				if (keepData && size > 0)
				{
					oldData = m_data;
					oldSize = m_size;
				}
				else
				{
					CUDA_SAFE_CALL((cudaFree(m_data)));
				}
				m_data = 0;
			}
			m_size = size;
			if (m_size)
			{
				void *tmp = 0;
				CUDA_SAFE_CALL((cudaMalloc(&tmp, m_size * sizeof(T))));
				m_data = reinterpret_cast<T*>(tmp);
			}

			if (oldData)
			{
				CUDA_SAFE_CALL(cudaMemcpy(m_data, oldData, ::min(uint32_t(oldSize), uint32_t(m_size)) * sizeof(T), cudaMemcpyDeviceToDevice));
				CUDA_SAFE_CALL((cudaFree(oldData)));
			}
		}
	}

	uint32_t size() const
	{
		return m_size;
	}

	uint32_t byteSize() const
	{
		return m_size * sizeof(T);
	}

	T* cudaPtr()
	{ 
		return m_data; 
	}

	const T* cudaPtr() const
	{ 
		return m_data; 
	}


	void set(uint8_t value, uint32_t count = ~uint32_t(0))
	{
		CUDA_SAFE_CALL(cudaMemset(m_data, value, min(uint32_t(count), uint32_t(m_size)) * sizeof(T)));
	}

	const T* beginHostMap(uint32_t count = ~uint32_t(0))
	{
		assert(!m_hostData);
		m_hostData = new T[min(uint32_t(count), uint32_t(m_size))];
		CUDA_SAFE_CALL(cudaMemcpy(m_hostData, m_data, min(uint32_t(count), uint32_t(m_size)) * sizeof(T), cudaMemcpyDeviceToHost));
		return m_hostData;
	}

	void endHostMap()
	{
		assert(m_hostData);
		delete [] m_hostData;
		m_hostData = 0;
	}

	void copy(const CudaBuffer<T> &b, uint32_t count)
	{
		assert(m_size >= count);
		cudaMemcpy(m_data, b.cudaPtr(), count * sizeof(T), cudaMemcpyDeviceToDevice);
	}

	void copyFromHost(const T *data, uint32_t count)
	{
		assert(m_size >= count);
		cudaMemcpy(m_data, data, count * sizeof(T), cudaMemcpyHostToDevice);
	}

	void copyToHost(T *hostData, uint32_t count, bool async = false, cudaStream_t stream = 0)
	{
		ASSERT(m_size >= count);
		ASSERT(!m_asyncInProgress);
		if (async)
		{
			m_asyncInProgress = true;
			cudaMemcpyAsync(hostData, m_data, count * sizeof(T), cudaMemcpyDeviceToHost, stream);
			cudaEventRecord(m_asyncCopyEvent, stream);
		}
		else
		{
			cudaMemcpy(hostData, m_data, count * sizeof(T), cudaMemcpyDeviceToHost);
		}
	}

	/**
	 * Ensure async copies have finished.
	 */
	void sync()
	{
		if (m_asyncInProgress)
		{
			cudaEventSynchronize(m_asyncCopyEvent);
			m_asyncInProgress = false;
		}
	}

	void operator=(const CudaBuffer &other)
	{
		resize(other.size(), true);
		copy(other, other.size());
	}

private:

	uint32_t m_size;
	cudaEvent_t m_asyncCopyEvent;
	bool m_asyncInProgress;
	T *m_data;
	T *m_hostData;
};

#endif // _CudaBuffer_h_
