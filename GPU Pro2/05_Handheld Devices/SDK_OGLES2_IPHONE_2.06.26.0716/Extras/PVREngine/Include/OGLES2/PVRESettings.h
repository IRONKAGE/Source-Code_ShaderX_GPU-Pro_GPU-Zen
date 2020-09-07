/******************************************************************************

 @File         PVRESettings.h

 @Title        PVREngine main header file for OGLES2 API

 @Version      

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent/OGLES2

 @Description  API independent settings for the PVREngine

******************************************************************************/

#ifndef PVRESETTINGS_H
#define PVRESETTINGS_H

#include "PVRTools.h"

namespace pvrengine
{

	/*!  constants to aid in API abstraction */

	/*!  clear constants */
	const int PVR_COLOUR_BUFFER		= GL_COLOR_BUFFER_BIT;
	const int PVR_DEPTH_BUFFER		= GL_DEPTH_BUFFER_BIT;

	/*!  general constants */
	enum ECullFaceMode
	{
		PVR_NONE				= GL_NONE,
		PVR_FRONT				= GL_FRONT,
		PVR_BACK				= GL_BACK,
		PVR_FRONT_AND_BACK	= GL_FRONT_AND_BACK
	};
	
		
	/*!***************************************************************************
	* @Class PVRESettings
	* @Brief 	API independent settings for the PVREngine.
	* @Description 	API independent settings for the PVREngine.
	*****************************************************************************/
	class PVRESettings
	{
	public:
		/*!***************************************************************************
		@Function			PVRESettings
		@Description		blank constructor.
		*****************************************************************************/
		PVRESettings();

		/*!***************************************************************************
		@Function			~PVRESettings
		@Description		destructor.
		*****************************************************************************/
		~PVRESettings();

		/*!***************************************************************************
		@Function			PVRESettings
		@Description		API initialisation code.
		*****************************************************************************/
		void Init();

		/*!***************************************************************************
		@Function			setBackColour
		@Input				cBackColour - clear colour
		@Description		sets the clear colour
		*****************************************************************************/
		void setBackColour(const unsigned int cBackColour);

		/*!***************************************************************************
		@Function			setBackColour
		@Input				fRed - red component of colour
		@Input				fGreen - red component of colour
		@Input				fBlue - red component of colour
		@Input				fAlpha - red component of colour
		@Description		sets the clear colour
		*****************************************************************************/
		void setBackColour(const VERTTYPE fRed, const VERTTYPE fGreen, const VERTTYPE fBlue, const VERTTYPE fAlpha);

		/*!***************************************************************************
		@Function			setBackColour
		@Input				v4BackColour - new clear colour
		@Description		sets the clear colour
		*****************************************************************************/
		void setBackColour(const PVRTVec4 v4BackColour);

		/*!***************************************************************************
		@Function			setBackColour
		@Input				fRed - red component of colour
		@Input				fGreen - red component of colour
		@Input				fBlue - red component of colour
		@Description		sets the clear colour
		*****************************************************************************/
		void setBackColour(const VERTTYPE fRed, const VERTTYPE fGreen, const VERTTYPE fBlue);

		/*!***************************************************************************
		@Function			setBackColour
		@Input				u32Red - red component of colour
		@Input				u32Green - red component of colour
		@Input				u32Blue - red component of colour
		@Input				u32Alpha - red component of colour
		@Description		sets the clear colour
		*****************************************************************************/
		void setBackColour(const unsigned int u32Red,
			const unsigned int u32Green,
			const unsigned int u32Blue,
			const unsigned int u32Alpha);

		/*!***************************************************************************
		@Function			setBackColour
		@Input				u32Red - red component of colour
		@Input				u32Green - red component of colour
		@Input				u32Blue - red component of colour
		@Description		sets the clear colour
		*****************************************************************************/
		void setBackColour(const unsigned int u32Red,
			const unsigned int u32Green,
			const unsigned int u32Blue);

		/*!***************************************************************************
		@Function			setClearFlags
		@Input				u32ClearFlags - see constants above
		@Description		sets which buffers that the clear operation affects
		*****************************************************************************/
		void setClearFlags(unsigned int u32ClearFlags);

		/*!***************************************************************************
		@Function			getClearFlags
		@Return				the current clear settnigs - see constants above
		@Description		sets which buffers that the clear operation affects
		*****************************************************************************/
		unsigned int getClearFlags();

		/*!***************************************************************************
		@Function			setDepthTest
		@Input				bDepth - true test, false don't
		@Description		switches the z depth test on or off
		*****************************************************************************/
		void setDepthTest(const bool bDepth);

		/*!***************************************************************************
		@Function			setDepthWrite
		@Input				bDepthWrite - true write, false don't
		@Description		sets whether rendering writes to the z depth buffer
		*****************************************************************************/
		void setDepthWrite(const bool bDepthWrite);

		/*!***************************************************************************
		 @Function			setBlend
		 @Input				bBlend - true blend, false don't
		 @Description		sets whether alpha blending is enabled according to the
		 current blend mode
		 *****************************************************************************/
		void setBlend(const bool bBlend);
		
		/*!***************************************************************************
		 @Function			setBlendFunc
		 @Input				eSource - Source value
		 @Input				eDest - Destination value
		 @Description		sets the blend function - values are the same for alpha and RGB
		 *****************************************************************************/
		void setBlendFunc(const EPODBlendFunc eSource,
			const EPODBlendFunc eDest);
		
		/*!***************************************************************************
		 @Function			setBlendFunc
		 @Input				eSourceRGB - RGB Source value
		 @Input				eSourceA - A Source value
		 @Input				eDestRGB - RGB Destination value
		 @Input				eDestA - A Destination value
		 @Description		sets the blend function
		 *****************************************************************************/
		void setBlendFuncSeparate(const EPODBlendFunc eSourceRGB,
		const EPODBlendFunc eDestRGB,
		const EPODBlendFunc eSourceA,
		const EPODBlendFunc eDestA);

		/*!***************************************************************************
		 @Function			setBlendOperation
		 @Input				eOp - The blend operation value
		 @Description		sets the blend function - values are the same for alpha and RGB
		 *****************************************************************************/
		void setBlendOperation(const EPODBlendOp eOp);
		
		/*!***************************************************************************
		 @Function			setBlendOperationSeparate
		 @Input				eOpRGB - RGB Source value
		 @Input				eOpA - A Source value
		 @Description		sets the blend function
		 *****************************************************************************/
		void setBlendOperationSeparate(const EPODBlendOp eOpRGB,
			const EPODBlendOp eOpA);

		/*!***************************************************************************
		@Function			setBlendColour
		@Input				v4BlendColour - blend colour to be set
		@Description		sets the blend colour to be used
		*****************************************************************************/
		void setBlendColour(const PVRTVec4 v4BlendColour);

		/*!***************************************************************************
		@Function			setCullFce
		@Input				bCullFace - true cull, false don't
		@Description		sets whether culling is enabled according to the current
		cull mode
		*****************************************************************************/
		void setCullFace(const bool bCullFace);

		/*!***************************************************************************
		@Function			setCullMode
		@Input				eMode - the cull mode
		@Description		sets the cull mode
		*****************************************************************************/
		void setCullMode(const ECullFaceMode eMode);

		/*!***************************************************************************
		@Function			Clear
		@Description		Performs a clear
		*****************************************************************************/
		void Clear();

		/*!***************************************************************************
		@Function			setClearFlags
		@Input				sPrint3d - reference to the Print3D class for this context
		@Input				u32Width - width of viewport
		@Input				u32Height - height of viewport
		@Input				bRotate - rotate for portrait flag
		@Description		Mandatory
		*****************************************************************************/
		bool InitPrint3D(CPVRTPrint3D& sPrint3d,
			const unsigned int u32Width,
			const unsigned int u32Height,
			const bool bRotate);

		/*!***************************************************************************
		@Function			getAPIName
		@Return				human readable string of the current API
		@Description		Returns a string containing the name of the current API:
		e.g. "OpenGL"
		*****************************************************************************/
		CPVRTString getAPIName();


		
		enum EStateFlag
		{
			eStateBackColour	= 1,
			eStateDepthTest		= 2,
			eStateBlendColour	= 4,
			eStateBlendOp		= 8,
			eStateDepthWrite	= 0x10,
			eStateBlendFunction = 0x20,
			eStateCullFace		= 0x40,
			eStateCullFaceMode	= 0x80,

		};

		/*!***************************************************************************
		@Function			setDirty
		@Input				eFlag - setting to mark dirty
		@Description		Marking dirty means that calls to this class will
		*****************************************************************************/	
		inline void setDirty(const EStateFlag eFlag)
		{
			m_u32DirtyFlags = m_u32DirtyFlags|eFlag;
		}

	protected:
		unsigned int		m_u32ClearFlags;	/*! current value to clear with */
		unsigned int		m_u32DirtyFlags;	/*! is a state dirty or clean */

		PVRTVec4		m_v4BackColour;			/*! background clear colour */
		PVRTVec4		m_v4BlendColour;		/*! blend colour */
		EPODBlendOp		m_eBlendOperationRGB,m_eBlendOperationA;		/*! blend operation */
		EPODBlendFunc	m_eBlendSourceRGB,m_eBlendDestRGB,m_eBlendSourceA,m_eBlendDestA;			/*! blend function */
		bool			m_bDepthTest;			/*! is depth testing on */
		bool			m_bDepthWrite;			/*! is depth writing on */
		bool			m_bCullFace;			/*! Face culling is on */
		ECullFaceMode	m_eCullFaceMode;		/*! Face culling mode */

		inline bool isDirty(const EStateFlag eFlag)
		{
			return (m_u32DirtyFlags&eFlag)!=0;
		}

		inline bool isClean(const EStateFlag eFlag)
		{
			return (m_u32DirtyFlags&eFlag)==0;
		}

		inline void setClean(const EStateFlag eFlag)
		{
			m_u32DirtyFlags  = m_u32DirtyFlags&(!eFlag);
		}


	};

}
#endif // PVRESETTINGS_H

/******************************************************************************
End of file (PVRESettings.h)
******************************************************************************/

