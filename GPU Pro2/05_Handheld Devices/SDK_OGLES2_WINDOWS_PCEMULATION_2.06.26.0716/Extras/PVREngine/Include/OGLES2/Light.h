/******************************************************************************

 @File         Light.h

 @Title        Light

 @Version      

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent

 @Description  Class to hold a Light with some convenient
               constructors/functions/operators

******************************************************************************/

#ifndef LIGHT_H
#define LIGHT_H

/******************************************************************************
Includes
******************************************************************************/

#include "PVRTools.h"


namespace pvrengine
{
	/******************************************************************************
	Enums
	******************************************************************************/
	enum ELightType{
		eLightTypePoint=0,
		eLightTypeDirectional,
		eLightTypeSpot,				// not suppoted yet
		eLightTypePODPoint,
		eLightTypePODDirectional,
//		eLightPODSpot				// Not supported by POD yet
		eNumLightTypes
	};

	/*!***************************************************************************
	* @Class Light
	* @Brief Class to hold a Light
	* @Description Class to hold a Light
	*****************************************************************************/
	class Light
	{
	public:
		/*!***************************************************************************
		@Function			Light
		@Description		blank constructor.
		*****************************************************************************/
		Light(){}

		/*!***************************************************************************
		@Function			Light
		@Input				v3Colour - colour of light
		@Input				eLightType - type of light
		@Description		blank constructor.
		*****************************************************************************/
		Light(const PVRTVec3& v3Colour, const ELightType eLightType)
			:m_v3Colour(v3Colour),m_eLightType(eLightType){}

		/*!***************************************************************************
		@Function			Light
		@Description		deconstructor.
		*****************************************************************************/
		virtual ~Light(){}

		/*!***************************************************************************
		@Function			shineLight
		@Input				i32Index - index of the light to shine
		@Description		sets up a hardware light (not functional in OpenGL)
		*****************************************************************************/
		virtual void shineLight(unsigned int i32Index)=0;

		/*!***************************************************************************
		@Function			getColour
		@Returns			the colour of the light
		@Description		retrieves the colour of this light
		*****************************************************************************/
		PVRTVec3 getColour(){return m_v3Colour;}

		/*!***************************************************************************
		@Function			getType
		@Return				type of the light
		@Description		gets the type of this light.
		*****************************************************************************/
		ELightType getType(){return m_eLightType;}

	protected:
		PVRTVec3 m_v3Colour;		/*!  colour of this light */
		ELightType m_eLightType;	/*!  type of light */

	};

	/*!***************************************************************************
	* @Class LightPoint
	* @Brief Class to hold a Point Light
	* @Description Class to hold a Point Light
	*****************************************************************************/
	class LightPoint : public Light
	{
	public:
		/*!***************************************************************************
		@Function			LightPoint
		@Input				v3Position - position of the new light
		@Input				v3Colour - colour of the light
		@Description		Constructor.
		*****************************************************************************/
		LightPoint(const PVRTVec3 v3Position, const PVRTVec3 v3Colour);

		/*!***************************************************************************
		@Function			LightPoint
		@Input				v4Position - position of the new light
		@Input				v3Colour - colour of the light
		@Description		Constructor.
		*****************************************************************************/
		LightPoint(const PVRTVec4 v4Position, const PVRTVec3 v3Colour);

		/*!***************************************************************************
		@Function			setPosition
		@Input				v4Position - position of the light
		@Description		Sets the position of the light.
		*****************************************************************************/
		void setPosition(const PVRTVec4 v4Position);

		/*!***************************************************************************
		@Function			setPosition
		@Input				v3Position - position of the light
		@Description		Sets the position of the light.
		*****************************************************************************/
		void setPosition(const PVRTVec3 v3Position);

		/*!***************************************************************************
		@Function			getPositionPVRTVec4
		@Return				position of the light
		@Description		gets the position of the light.
		*****************************************************************************/
		PVRTVec4 getPositionPVRTVec4(){return m_v4Position;}

		/*!***************************************************************************
		@Function			getPositionPVRTVec3
		@Return				position of the light
		@Description		gets the position of the light.
		*****************************************************************************/
		PVRTVec3 getPositionPVRTVec3(){return PVRTVec3(m_v4Position.x,m_v4Position.y,m_v4Position.z);}

		/*!***************************************************************************
		@Function			shineLight
		@Input				u32Index - index of the light to shine
		@Description		sets up a hardware light (not functional in OpenGL)
		*****************************************************************************/
		void shineLight(unsigned int u32Index);

	protected:
		PVRTVec4 m_v4Position;		/*! position of this light */
	};

	/*!***************************************************************************
	* @Class LightDirectional
	* @Brief Class to hold a Directional Light
	* @Description Class to hold a Directional Light
	*****************************************************************************/
	class LightDirectional : public Light
	{
	public:
		/*!***************************************************************************
		@Function			LightDirectional
		@Input				v3Direction - direction of the new light
		@Input				v3Colour - colour of the light
		@Description		Constructor.
		*****************************************************************************/
		LightDirectional(const PVRTVec3 v3Direction, const PVRTVec3 v3Colour);

		/*!***************************************************************************
		@Function			LightDirectional
		@Input				v4Direction - direction of the new light
		@Input				v3Colour - colour of the light
		@Description		Constructor.
		*****************************************************************************/
		LightDirectional(const PVRTVec4 v4Direction, const PVRTVec3 v3Colour);

		/*!***************************************************************************
		@Function			setDirection
		@Input				v3Direction - direction of the light
		@Description		Sets the direction of the light.
		*****************************************************************************/
		void setDirection(const PVRTVec3 v3Direction);

		/*!***************************************************************************
		@Function			setDirection
		@Input				v4Direction - direction of the light
		@Description		Sets the direction of the light.
		*****************************************************************************/
		void setDirection(const PVRTVec4 v4Direction);

		/*!***************************************************************************
		@Function			getDirectionPVRTVec4
		@Return				direction of the light
		@Description		gets the direction of the light.
		*****************************************************************************/
		PVRTVec4 getDirectionPVRTVec4(){return m_v4Direction;}

		/*!***************************************************************************
		@Function			getDirectionPVRTVec3
		@Return				direction of the light
		@Description		gets the direction of the light.
		*****************************************************************************/
		PVRTVec3 getDirectionPVRTVec3(){return PVRTVec3(m_v4Direction.x,m_v4Direction.y,m_v4Direction.z);}

		/*!***************************************************************************
		@Function			shineLight
		@Input				u32Index - index of the light to shine
		@Description		sets up a hardware light (not functional in OpenGL)
		*****************************************************************************/
		void shineLight(unsigned int u32Index);

	protected:
		PVRTVec4 m_v4Direction;		/*! direction of this light */
	};


	/*!***************************************************************************
	* @Class LightPODDirectional
	* @Brief Class to hold a POD Directional Light
	* @Description Class to hold a POD Directional Light
	*****************************************************************************/
	class LightPODDirectional : public Light
	{
	public:
		/*!***************************************************************************
		@Function			LightPODDirectional
		@Input				pModelPOD - POD model from which light should come
		@Input				iIndex - index into POD of this light
		@Description		Constructor.
		*****************************************************************************/
		LightPODDirectional(CPVRTModelPOD* pModelPOD, const int iIndex):
	  		  Light(pModelPOD->pLight[iIndex].pfColour,eLightTypePODDirectional),m_psModelPOD(pModelPOD),m_iIndex(iIndex){}

		/*!***************************************************************************
		@Function			getDirectionPVRTVec4
		@Return				direction of the light
		@Description		gets the direction of the light.
		*****************************************************************************/
		PVRTVec4 getDirectionPVRTVec4(){return m_psModelPOD->GetLightDirection(m_iIndex);}

		/*!***************************************************************************
		@Function			getDirectionPVRTVec3
		@Return				direction of the light
		@Description		gets the direction of the light.
		*****************************************************************************/
		PVRTVec3 getDirectionPVRTVec3(){return m_psModelPOD->GetLightDirection(m_iIndex);}

		/*!***************************************************************************
		@Function			shineLight
		@Input				u32Index - index of the light to shine
		@Description		sets up a hardware light (not functional in OpenGL)
		*****************************************************************************/
		void shineLight(unsigned int u32Index);

		/*!***************************************************************************
		@Function			isPODLight
		@Return				whether light originated from a POD
		@Description		determines whether light originated from a POD
		*****************************************************************************/
		CPVRTModelPOD* getModelPOD(){return m_psModelPOD;}

	protected:
		CPVRTModelPOD *m_psModelPOD;		/*!  scene from which PDO originates */
		int m_iIndex;						/*! index of light in scene */
	};

	/*!***************************************************************************
	* @Class LightPODPoint
	* @Brief Class to hold a POD Point Light
	* @Description Class to hold a POD Point Light
	*****************************************************************************/
	class LightPODPoint : public Light
	{
	public:
		/*!***************************************************************************
		@Function			LightPODPoint
		@Input				pModelPOD - POD model from which light should come
		@Input				iIndex - index into POD of this light
		@Description		Constructor.
		*****************************************************************************/
		LightPODPoint(CPVRTModelPOD* pModelPOD, const int iIndex):
		  Light(pModelPOD->pLight[iIndex].pfColour,eLightTypePODPoint),m_psModelPOD(pModelPOD),m_iIndex(iIndex){}

		/*!***************************************************************************
		@Function			getDirectionPVRTVec4
		@Return				direction of the light
		@Description		gets the direction of the light.
		*****************************************************************************/
		PVRTVec4 getPositionPVRTVec4(){return m_psModelPOD->GetLightPosition(m_iIndex);}

		/*!***************************************************************************
		@Function			getDirectionPVRTVec3
		@Return				direction of the light
		@Description		gets the direction of the light.
		*****************************************************************************/
		PVRTVec3 getPositionPVRTVec3(){return m_psModelPOD->GetLightPosition(m_iIndex);}

		/*!***************************************************************************
		@Function			shineLight
		@Input				u32Index - index of the light to shine
		@Description		sets up a hardware light (not functional in OpenGL)
		*****************************************************************************/
		void shineLight(unsigned int u32Index);

		/*!***************************************************************************
		@Function			isPODLight
		@Return				whether light originated from a POD
		@Description		determines whether light originated from a POD
		*****************************************************************************/
		CPVRTModelPOD* getModelPOD(){return m_psModelPOD;}

	protected:
		CPVRTModelPOD *m_psModelPOD;		/*!  scene from which PDO originates */
		int m_iIndex;						/*! index of light in scene */
	};

}

#endif // LIGHT_H

/******************************************************************************
End of file (Light.h)
******************************************************************************/

