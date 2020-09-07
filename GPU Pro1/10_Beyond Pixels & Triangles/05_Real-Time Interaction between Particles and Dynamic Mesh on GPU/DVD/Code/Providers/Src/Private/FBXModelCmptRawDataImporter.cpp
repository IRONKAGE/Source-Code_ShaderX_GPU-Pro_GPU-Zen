#include "FBXPrecompiled.h"

#include "Common/Src/auto_array.h"

#include "Math/Src/Operations.h"

#include "Providers.h"
#include "SkeletonRawDataImporterProvider.h"
#include "FBXSkeletonRawDataImporter.h"

#include "RawVertexData.h"
#include "TextureNamesSet.h"
#include "ModelCmptRawData.h"

#include "AnimationTrack.h"
#include "RawSkeletonData.h"

#include "FBXHelpers.cpp.h"

#include "FBXCommon.h"

#include "FBXModelCmptRawDataImporter.h"

namespace Mod
{

	using namespace FBX;
	using namespace FBXFILESDK_NAMESPACE;
	using namespace Math;

	//------------------------------------------------------------------------
	// use this namespace to avoid anonymous namespace VS debugging "features"

	namespace FBXVertexTransformation
	{
		struct VertexExtraData
		{
			float3 normal;
			float3 uv;

			// don't init normal as it is always present
			VertexExtraData() : uv( ZERO_3 )
			{

			}

			bool operator == ( const VertexExtraData& vedata ) const
			{
				float3 nd	= normal - vedata.normal;
				float3 uvd	= uv - vedata.uv;
				return dot( nd, nd ) + dot( uvd, uvd ) <= std::f;

ommon ) dot( ruuonimexExtraD"	== SkeletonRawDatae conData rincluders.h"
#include "feak.h"
#includes pm
	ndingTet
l�tuocompVertextraDKvd, (  uv -{aop o ommon/Sroviders.h"
#& vedata a al Vertex normal - vedata.normal;
	usingaees��it3 V�aertex�Meak.d nation
	{
E�turawing "f�
fotionDatalPh�eda�vP;
	us!= avoider h"�rust
			{
			��mationFtaIrack.h"
#includes ud>#idat
lܘBX
	//SFS�S �
7amexto	/rov;

			// don
	{
		sVS;

	using namespace  uvgaFSmportex �Co 
:eodet VertexE�turam	baertexExtraDatatRi	{atforust
			aeera(tReit Vertexer- uDtraetRed.h"
#includetion
l�