#ifndef __HashVector_h_
#define __HashVector_h_

#include "IntTypes.h"
#include "HashFunction.h"
#include <memory>
#include <utility>
/** 
 * A hash vector is called thus because it combines features from both hash tables and
 * vectors. It has the performance characteristics of a hash table, but inserted data
 * ends up in an (insertion)ordered array and can be random accessed.
 */
template <typename KEY_T, typename OBJECT_T, typename HASHFUN_T = HashFunction<KEY_T> >
class HashVector
{
public:
  struct Data
  {
    Data(const KEY_T &_key, const OBJECT_T &_object) :
      key(_key),
      object(_object)
    {
    }
    KEY_T key;
    OBJECT_T object;

	bool operator== (const Data& data) const
	{
		return HASHFUN_T().isEqual(key, data.key) &&
			object == data.object;
	}
  };

  typedef Data *iterator;
  typedef const Data *const_iterator;

  typedef Data& reference;
  typedef const Data& const_reference;

  HashVector(size_t capacity, size_t bucketsPerElement = 4) :
      m_buckets(0),
      m_data(0),
      m_numBuckets(capacity * bucketsPerElement),
      m_capacity(capacity),
      m_size(0)
  {
    m_data = new uint8_t[m_capacity * sizeof(Data)];
    m_bucketIndexOfData = new int[m_capacity];
    m_buckets = new int[m_numBuckets];
    for(size_t i = 0; i < m_numBuckets; ++i)
    {
      m_buckets[i] = -1;
    }
  }

  ~HashVector()
  {
    clear();

    delete [] m_data;
    delete [] m_bucketIndexOfData;
    delete [] m_buckets;
  }

  void insert(const KEY_T &key, const OBJECT_T &object)
  {
    HashID hash = HASHFUN_T().hash(key);
    size_t bucketIndex = hash % m_numBuckets;
    for(size_t i = bucketIndex; i < bucketIndex + m_numBuckets; ++i)
    {
      size_t index = i % m_numBuckets;
      // empty buckets are either -1 (never used) or -2 previously used.
      if(m_buckets[index] < 0)
      {
        m_buckets[index] = int(m_size);
        new(dataAt(m_size)) Data(key, object);
        m_bucketIndexOfData[m_size] = int(index);
        ++m_size;
        break;
      }
    }
  }

  void insert(const std::pair<KEY_T, OBJECT_T> & thing)
  {
    HashID hash = HASHFUN_T().hash(thing.first);
    size_t bucketIndex = hash % m_numBuckets;
    for(size_t i = bucketIndex; i < bucketIndex + m_numBuckets; ++i)
    {
      size_t index = i % m_numBuckets;
      // empty buckets are either -1 (never used) or -2 previously used.
      if(m_buckets[index] < 0)
      {
        m_buckets[index] = int(m_size);
        new(dataAt(m_size)) Data(thing.first, thing.second);
        m_bucketIndexOfData[m_size] = int(index);
        ++m_size;
        break;
      }
    }
  }

  iterator insert( const iterator&, const_reference thing )
  {
	  // provide compatibility with standard containers, specifically the
	  // insert_iterator()
	  insert( thing.key, thing.object );

	  return iterator(0);
  }

  void remove(const KEY_T &key)
  {
    HashID hash = HASHFUN_T().hash(key);
    size_t bucketIndex = hash % m_numBuckets;
    for(size_t i = bucketIndex; i < bucketIndex + m_numBuckets; ++i)
    {
      const size_t bucketIndex = i % m_numBuckets;
      const int dataIndex = m_buckets[bucketIndex];
      // empty buckets are either -1, never used, or -2, previously used.
      if(dataIndex >= 0)
      {
        Data* data = dataAt(dataIndex);
        if(HASHFUN_T().isEqual(data->key, key))
        {
          // flag bucket as empty and reusable.
          m_buckets[bucketIndex] = -2;
          // if there are more than 1 item in the vector, and we're not removing the last one
          // we need to compact data into the hole in the array.
          if(m_size > 1 && dataIndex != m_size - 1)
          {
            *data = *dataAt(m_size - 1);
            // and then update the hash bucket indexing the last item, to index the new location
            m_buckets[m_bucketIndexOfData[m_size - 1]] = int(dataIndex);
            // update the bucket index of the moved item, to index the new data's bucket.
            m_bucketIndexOfData[dataIndex] = m_bucketIndexOfData[m_size - 1];
          }
          // now the last item is either copied or should die
          dataAt(m_size - 1)->~Data();
          --m_size;

          return;
        }
      }
      else if(dataIndex == -1)
      {
        return;
      }
    }
  }

  iterator find(const KEY_T &key)
  {
    HashID hash = HASHFUN_T().hash(key);
    size_t bucketIndex = hash % m_numBuckets;
    for(size_t i = bucketIndex; i < bucketIndex + m_numBuckets; ++i)
    {
      int dataIndex = m_buckets[i % m_numBuckets];
      // empty buckets are either -1, never used, or -2, previously used.
      if(dataIndex >= 0)
      {
        Data* data = dataAt(dataIndex);
        if(HASHFUN_T().isEqual(data->key, key))
        {
          return data;
        }
      }
      else if(dataIndex == -1)
      {
        return end();
      }
    }

    return end();
  }

  size_t size() const { return m_size; }

  const_iterator begin() const { return dataAt(0); }
  iterator begin() { return dataAt(0); }

  const_iterator end() const { return dataAt(m_size); }
  iterator end() { return dataAt(m_size); }

  void clear()
  {
    for(size_t i = 0; i < m_numBuckets; ++i)
    {
      m_buckets[i] = -1;
    }

    for(size_t i = 0; i < m_size; ++i)
    {
      dataAt(i)->~Data();
    }
    m_size = 0;
  }
private: // data

  Data* dataAt(size_t index)
  {
    return reinterpret_cast<Data*>(m_data + sizeof(Data) * index);
  }

  const Data* dataAt(size_t index) const
  {
    return reinterpret_cast<Data*>(m_data + sizeof(Data) * index);
  }
  void operator=(const HashVector&);

  int *m_buckets; // [m_numBuckets] hash bucket array with indices into the data array.
  uint8_t *m_data; // [m_capacity] continuous array of data items, only the m_size first are constructed.
  int *m_bucketIndexOfData; //[m_capacity] continuous array of indices of buckets, one for each data item, telling us where the item was hashed to.
  size_t m_numBuckets; // number of elements in the buckets array >= m_capacity
  size_t m_capacity; // max number of data items that can be inserted.
  size_t m_size; // current number of data items inserted.
};

#endif // __HashVector_h_
