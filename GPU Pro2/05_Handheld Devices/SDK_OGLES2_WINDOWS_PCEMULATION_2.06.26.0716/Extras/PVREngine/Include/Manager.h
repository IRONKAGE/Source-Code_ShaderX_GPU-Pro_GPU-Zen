/******************************************************************************

 @File         Manager.h

 @Title        A simple light manager for use with PVREngine

 @Version      

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent - OGL/OGLES/OGLES2 Specific at the moment

 @Description  A class for holding information about lights

******************************************************************************/

#ifndef _MANAGER_H_
#define _MANAGER_H_

/******************************************************************************
Includes
******************************************************************************/

#include "PVRTools.h"
#include "dynamicArray.h"

namespace pvrengine
{

	/*!***************************************************************************
	* @Class Manager
	* @Brief A class for holding information about lights
	* @Description A class for holding information about lights
	*****************************************************************************/
	template<typename T>
	class Manager
	{
	public:
		/*!***************************************************************************
		@Function			Manager
		@Description		blank constructor.
		*****************************************************************************/
		Manager(){}

		/*!***************************************************************************
		@Function			~Manager
		@Description		destructor.
		*****************************************************************************/
		virtual ~Manager(){}

		/*!***************************************************************************
		@Function			Manager
		@Input				i32Capacity
		@Description		constructor taking specific initial capacity value.
		*****************************************************************************/
		Manager(int i32Capacity)
		{
			m_daElements.expandToSize(i32Capacity);
		}

		/*!***************************************************************************
		@Function			add
		@Input				pElement - an element of type T*
		@Return				handle to this element
		@Description		add an element to this manager.
		*****************************************************************************/
		void add(T *pElement)
		{
			m_daElements.append(pElement);
		}

		/*!***************************************************************************
		@Function			get
		@Input				i32Index	index of requested object
		@Return			bool success
		@Description		sets the path to the output file.
		*****************************************************************************/
		T*	get(int i32Index) const
		{
			return m_daElements[i32Index];
		}

		/*!***************************************************************************
		@Function			get
		@Return				size of store
		@Description		sets the path to the output file.
		*****************************************************************************/
		int	getSize() const
		{
			return m_daElements.getSize();
		}

		/*!***************************************************************************
		@Function			getAll
		@Return				the dynamic array of all elements in the manager
		@Description		Direct access to repository.
		*****************************************************************************/
		dynamicArray<T*>*	getAll(){ return &m_daElements;}

		/*!***************************************************************************
		@Function			sort
		@Description		Sorts elements
		*****************************************************************************/
		void sort()
		{
			m_daElements.bubbleSortPointers();
		}

	protected:
		dynamicPointerArray<T> m_daElements;	/*! the store */
	};

}
#endif // _MANAGER_H_

/******************************************************************************
End of file (Manager.h)
******************************************************************************/

