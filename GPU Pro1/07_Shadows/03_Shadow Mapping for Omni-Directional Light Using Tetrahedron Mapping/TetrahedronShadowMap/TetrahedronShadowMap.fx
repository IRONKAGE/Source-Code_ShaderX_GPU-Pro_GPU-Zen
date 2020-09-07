//-----------------------------------------------------------------------------
// File: TetrahedronShadowMap.fx
//
// Desc: Effect file for Tetrahedron shadow map.
//
// Copyright (c) Hung-Chien Liao. All rights reserved.
//-----------------------------------------------------------------------------


#define SMAP_SIZE 512


#define SHADOW_EPSILON 0.00005f

float4x4 g_mWorldViewProj;		// Tranform from object to camera view projection space
float4x4 g_mWorldLightView;		// Transform from object to light view space
float4x4 g_mViewToLightProj;	// Transform from view space to light projection space
float4x4 g_mTexTransform[6];	// Transform point into final perspective shadow map sapace
texture  g_txDiffuse;			// diffuse texture
texture  g_txShadowFront;		// front side shadow map
texture  g_txShadowBack;		// back side shadow map
texture  g_txCubeShadow;		// cube shadow map
texture  g_txCubeToTSM;			// The look up texture from cube to tetrahedron coordinate
float4   g_vLightDiffuse = float4( 1.0f, 1.0f, 1.0f, 1.0f );  // Light diffuse color
float4   g_vLightAmbient = float4( 0.3f, 0.3f, 0.3f, 1.0f );  // Use an ambient light of 0.3
float4	 g_vLightAttenuation = float4(0.0f, 0.0f, 0.002f, 0.0f);
float	 g_fLightRangeSquare;
float4x4 g_mTSMFaceCenter = {0.0f,			0.0f,			-0.81649655f,	0.81649655f,
							-0.57735026f,	-0.57735026f,	0.57735026f,	0.57735026f,
							0.81649661f,	-0.81649661f,	0.0f,			0.0f,
							0.0f,			0.0f,			0.0f,			0.0f};
float3	g_vFace1Center = { 0.0f, -0.57735026f, 0.81649661f};
float3	g_vFace2Center = { 0.0f, -0.57735026f, -0.81649661f};
float3	g_vFace3Center = { -0.81649655f, 0.57735026f, 0.0f};
float3	g_vFace4Center = { 0.81649655f, 0.57735026f, 0.0f};
float4	g_vFace14Test = { -0.57735026f, 0.81649661f, -0.81649655f, -0.57735026f};

struct PS_OUTPUT
{
   float4 Color:	COLOR0;
   float Depth:		DEPTH;
};

sampler2D g_samDiffuse = sampler_state
{
    Texture = <g_txDiffuse>;
    MinFilter = Point;
    MagFilter = Linear;
    MipFilter = Linear;
};

sampler2D g_samShadowFront = sampler_state
{
    Texture = <g_txShadowFront>;
    MinFilter = Linear;
    MagFilter = Linear;
    MipFilter = None;
    AddressU = WRAP;	// Need wrap for look up shadow map
    AddressV = WRAP;	// Need wrap for look up shadow map
};

sampler2D g_samShadowBack = sampler_state
{
    Texture = <g_txShadowBack>;
    MinFilter = Linear;
    MagFilter = Linear;
    MipFilter = None;
    AddressU = Clamp;
    AddressV = Clamp;
};

samplerCUBE g_samCubeShadow = sampler_state
{
    Texture = <g_txCubeShadow>;
    MinFilter = Linear;
    MagFilter = Linear;
    MipFilter = None;
    AddressU = Clamp;
    AddressV = Clamp;
};

samplerCUBE g_samCubeToTSM = sampler_state
{
	Texture = <g_txCubeToTSM>;
	AddressU  = WRAP;        
    AddressV  = WRAP;
    AddressW  = CLAMP;
    MIPFILTER = NONE;
    MINFILTER = LINEAR;
    MAGFILTER = LINEAR;
};

//-----------------------------------------------------------------------------
// Vertex Shader: VSPointTSM
// Desc: Process vertex for scene with Tetrahedron shadow map
//-----------------------------------------------------------------------------
void VSPointTSM(float4 iPos : POSITION,	// input vertex position in object space
	float3 iNormal   : NORMAL,			// input vertex normal
	float2 iTexCoord : TEXCOORD0,		// input texture coordinate
	out float4 oPos : POSITION,			// output vertex position in clip space
	out float2 oTexCoord0 : TEXCOORD0,	// output texture coordinate for diffuse map
	out float3 oNormal : TEXCOORD1,		// output the vertex normal in light view space
	out float3 oVertexPos :TEXCOORD2,	// output the vertex position in light view space
	out float4 oSMTexCoord0 : TEXCOORD3,
	out float4 oSMTexCoord1 : TEXCOORD4,
	out float4 oSMTexCoord2 : TEXCOORD5,
	out float4 oSMTexCoord3 : TEXCOORD6)	// Shadow map texture coordinate
{
	// copy texture coordinates
	oTexCoord0 = iTexCoord;

	// transform position to clip space
	oPos = mul(iPos, g_mWorldViewProj);
			
	// Transform unit vertex position from object space to light view space
	oVertexPos.xyz = mul(iPos, g_mWorldLightView);
	float4 vVertexPos = float4(oVertexPos.xyz, 1.0f);
	oSMTexCoord0 = mul(vVertexPos, g_mTexTransform[0]);
	oSMTexCoord1 = mul(vVertexPos, g_mTexTransform[1]);
	oSMTexCoord2 = mul(vVertexPos, g_mTexTransform[2]);
	oSMTexCoord3 = mul(vVertexPos, g_mTexTransform[3]);
	
	// Transform the normal to light view space
	oNormal = mul(iNormal, g_mWorldLightView);
}


//-----------------------------------------------------------------------------
// Pixel Shader: PSPointTSM
// Desc: Process pixel for scene with Tetrahedron shadow map
//-----------------------------------------------------------------------------
float4 PSPointTSM(float2 iTexCoord0 : TEXCOORD0,	// input texture coordinate for diffuse map
	float3 iNormal : TEXCOORD1,		// input the vertex normal in light view space
	float3 iVertexPos : TEXCOORD2,	// input the vertex position in light view space
	float4 iSMTexCoord0 : TEXCOORD3,// Shadow map texture coordinate
	float4 iSMTexCoord1 : TEXCOORD4,
	float4 iSMTexCoord2 : TEXCOORD5,
	float4 iSMTexCoord3 : TEXCOORD6) : COLOR 
{
	// Determine the distance from the light to the vertex and the direction
	float3 vLightDir = -iVertexPos;
	float fDistance = length(vLightDir);
	vLightDir = vLightDir / fDistance;
	float fNdotL = dot(vLightDir, iNormal);
	// Compute the per-pixel distance based attenuation
	float fAttenuation = clamp( 0, 1, 1 / (g_vLightAttenuation.x +
		g_vLightAttenuation.y * fDistance + g_vLightAttenuation.z * fDistance * fDistance) );
	
	float4 vFaceDeter = mul(float4(iVertexPos.xyz, 0.0f), g_mTSMFaceCenter);
	float fMax = max(max(vFaceDeter.x, vFaceDeter.y), max(vFaceDeter.z, vFaceDeter.w));
	float4 shadowTexCoordDepth;
	if (vFaceDeter.x == fMax)
		shadowTexCoordDepth = iSMTexCoord0;
	else if (vFaceDeter.y == fMax)
		shadowTexCoordDepth = iSMTexCoord1;
	else if (vFaceDeter.z == fMax)
		shadowTexCoordDepth = iSMTexCoord2;
	else
		shadowTexCoordDepth = iSMTexCoord3;
		
	float shadow = (shadowTexCoordDepth.z / shadowTexCoordDepth.w <= tex2Dproj(g_samShadowFront, shadowTexCoordDepth));
		
	// Texture color * Lighting(Diffuse * Attenuation) Illumination
	return tex2D(g_samDiffuse, iTexCoord0) * ((fNdotL * g_vLightDiffuse) * fAttenuation * shadow);
}

//-----------------------------------------------------------------------------
// Pixel Shader: PSPointTSMHardware
// Desc: Process pixel for scene with Tetrahedron shadow map (Hardware Shadow Map)
//-----------------------------------------------------------------------------
float4 PSPointTSMHardware(float2 iTexCoord0 : TEXCOORD0,	// input texture coordinate for diffuse map
	float3 iNormal : TEXCOORD1,		// input the vertex normal in light view space
	float3 iVertexPos : TEXCOORD2,	// input the vertex position in light view space
	float4 iSMTexCoord0 : TEXCOORD3,// Shadow map texture coordinate
	float4 iSMTexCoord1 : TEXCOORD4,
	float4 iSMTexCoord2 : TEXCOORD5,
	float4 iSMTexCoord3 : TEXCOORD6) : COLOR 
{
	// Determine the distance from the light to the vertex and the direction
	float3 vLightDir = -iVertexPos;
	float fDistance = length(vLightDir);
	vLightDir = vLightDir / fDistance;
	float fNdotL = dot(vLightDir, iNormal);
	// Compute the per-pixel distance based attenuation
	float fAttenuation = clamp( 0, 1, 1 / (g_vLightAttenuation.x +
		g_vLightAttenuation.y * fDistance + g_vLightAttenuation.z * fDistance * fDistance) );
	
	float4 vFaceDeter = mul(float4(iVertexPos.xyz, 0.0f), g_mTSMFaceCenter);
	float fMax = max(max(vFaceDeter.x, vFaceDeter.y), max(vFaceDeter.z, vFaceDeter.w));
	float4 shadowTexCoordDepth;
	if (vFaceDeter.x == fMax)
		shadowTexCoordDepth = iSMTexCoord0;
	else if (vFaceDeter.y == fMax)
		shadowTexCoordDepth = iSMTexCoord1;
	else if (vFaceDeter.z == fMax)
		shadowTexCoordDepth = iSMTexCoord2;
	else
		shadowTexCoordDepth = iSMTexCoord3;
		
	float shadow = tex2Dproj(g_samShadowFront, shadowTexCoordDepth);
		
	// Texture color * Lighting(Diffuse * Attenuation) Illumination
	return tex2D(g_samDiffuse, iTexCoord0) * ((fNdotL * g_vLightDiffuse) * fAttenuation * shadow);
}

//-----------------------------------------------------------------------------
// Vertex Shader: VSPointTSMLook
// Desc: Process vertex for scene with Tetrahedron shadow map and Lookup texture
//-----------------------------------------------------------------------------
void VSPointTSMLook(float4 iPos : POSITION,	// input vertex position in object space
	float3 iNormal   : NORMAL,			// input vertex normal
	float2 iTexCoord : TEXCOORD0,		// input texture coordinate
	out float4 oPos : POSITION,			// output vertex position in clip space
	out float2 oTexCoord0 : TEXCOORD0,	// output texture coordinate for diffuse map
	out float3 oNormal : TEXCOORD1,		// output the vertex normal in light view space
	out float3 oVertexPos :TEXCOORD2)	// output the vertex position in light view space
{
	// copy texture coordinates
	oTexCoord0 = iTexCoord;

	// transform position to clip space
	oPos = mul(iPos, g_mWorldViewProj);
			
	// Transform unit vertex position from object space to light view space
	oVertexPos.xyz = mul(iPos, g_mWorldLightView);
	
	// Transform vertex normal from object space to light view space
	oNormal = mul(iNormal, g_mWorldLightView);	
}

//-----------------------------------------------------------------------------
// Pixel Shader: PSPointTSMLook
// Desc: Process pixel for scene with Tetrahedron shadow map and Lookup texture
//-----------------------------------------------------------------------------
float4 PSPointTSMLook(float2 iTexCoord0 : TEXCOORD0,	// input texture coordinate for diffuse map
	float3 iNormal : TEXCOORD1,				// input the vertex normal in light view space
	float3 iVertexPos : TEXCOORD2) : COLOR	// output the vertex position in light view space
{
	// Determine the distance from the light to the vertex and the direction
	float3 vLightDir = -iVertexPos;
	float fDistance = length(vLightDir);
	float fDistanceSquare = fDistance * fDistance;
	vLightDir = vLightDir / fDistance;
	float fNdotL = dot(vLightDir, iNormal);
	// Compute the per-pixel distance based attenuation
	float fAttenuation = clamp( 0, 1, 1 / (g_vLightAttenuation.x +
		g_vLightAttenuation.y * fDistance + g_vLightAttenuation.z * fDistanceSquare) );
	
	float4 shadowMapCoord = texCUBE(g_samCubeToTSM, iVertexPos);
	float3 vAbsVertexPos = abs(iVertexPos);
	//float3 vTestVertexPos = float3(vAbsVertexPos.x, iVertexPos.y, vAbsVertexPos.z);	// The old way	
	float4 vTestVertexPos = float4(iVertexPos.y, vAbsVertexPos.z, vAbsVertexPos.x, iVertexPos.y);	// The faster way
	//if (dot(vTestVertexPos, g_vFace1Center) < dot(vTestVertexPos, g_vFace4Center))	// The old way
	if (dot(vTestVertexPos, g_vFace14Test) < 0)														// The faster way
		shadowMapCoord.xy = shadowMapCoord.zw;
	float shadow = (fDistanceSquare / g_fLightRangeSquare) <= tex2D(g_samShadowFront, shadowMapCoord.xy);
		
	// Texture color * Lighting(Diffuse * Attenuation) Illumination
	return tex2D(g_samDiffuse, iTexCoord0) * ((fNdotL * g_vLightDiffuse) * fAttenuation * shadow);
}

//-----------------------------------------------------------------------------
// Pixel Shader: PSPointTSMLookHardware
// Desc: Process pixel for scene with Tetrahedron shadow map (Hardware Shadow Map) and Lookup texture
//-----------------------------------------------------------------------------
float4 PSPointTSMLookHardware(float2 iTexCoord0 : TEXCOORD0,	// input texture coordinate for diffuse map
	float3 iNormal : TEXCOORD1,				// input the vertex normal in light view space
	float3 iVertexPos : TEXCOORD2) : COLOR	// output the vertex position in light view space
{
	// Determine the distance from the light to the vertex and the direction
	float3 vLightDir = -iVertexPos;
	float fDistance = length(vLightDir);
	float fDistanceSquare = fDistance * fDistance;
	vLightDir = vLightDir / fDistance;
	float fNdotL = dot(vLightDir, iNormal);
	// Compute the per-pixel distance based attenuation
	float fAttenuation = clamp( 0, 1, 1 / (g_vLightAttenuation.x +
		g_vLightAttenuation.y * fDistance + g_vLightAttenuation.z * fDistanceSquare) );
	
	float3 vAbsVertexPos = abs(iVertexPos);
	float4 vTestVertexPos = float4(iVertexPos.y, vAbsVertexPos.z, vAbsVertexPos.x, iVertexPos.y);
	float4 shadowMapCoord = texCUBE(g_samCubeToTSM, iVertexPos);
	if (dot(vTestVertexPos, g_vFace14Test) < 0)
		shadowMapCoord.xy = shadowMapCoord.zw;
		
	float4 shadowTexCoordDepth = float4(shadowMapCoord.xy, fDistanceSquare / g_fLightRangeSquare, 1.0f);
	float shadow = tex2Dproj(g_samShadowFront, shadowTexCoordDepth);
		
	// Texture color * Lighting(Diffuse * Attenuation) Illumination
	return tex2D(g_samDiffuse, iTexCoord0) * ((fNdotL * g_vLightDiffuse) * fAttenuation * shadow);
}

//-----------------------------------------------------------------------------
// Vertex Shader: VSPointDSM
// Desc: Process vertex for scene with Dual-paraboloid shadow map
//-----------------------------------------------------------------------------
// Vertex Shader for point light with specular, normal mapping and Cube shadow map
void VSPointDSM(float4 iPos : POSITION,	// input vertex position in object space
	float3 iNormal   : NORMAL,				// input vertex normal
	float2 iTexCoord : TEXCOORD0,			// input texture coordinate
	out float4 oPos : POSITION,			// output vertex position in clip space
	out float2 oTexCoord0 : TEXCOORD0,	// output texture coordinate for diffuse map
	out float3 oNormal : TEXCOORD1,		// output the vertex normal in light view space
	out float3 oVertexPos :TEXCOORD2,	// output the vertex position in light view space
	out float4 oSMTexCoord : TEXCOORD3)	// Shadow map texture coordinate(xy: front side, zw:back side)
{
	// copy texture coordinates
	oTexCoord0 = iTexCoord;

	// transform position to clip space
	oPos = mul(iPos, g_mWorldViewProj);
			
	// Transform unit vertex position from object space to light view space
	oVertexPos.xyz = mul(iPos, g_mWorldLightView);
	
	oNormal = mul(iNormal, g_mWorldLightView);
	
	float3 normalPos = normalize(oVertexPos.xyz);
	oSMTexCoord.x = (1 + (normalPos.x / (1 + normalPos.z))) * 0.5; // front
	oSMTexCoord.y = (1 - (normalPos.y / (1 + normalPos.z))) * 0.5; // front
	oSMTexCoord.z = (1 + (normalPos.x / (1 - normalPos.z))) * 0.5; // back
	oSMTexCoord.w = (1 - (normalPos.y / (1 - normalPos.z))) * 0.5; // back
}

//-----------------------------------------------------------------------------
// Pixel Shader: PSPointDSM
// Desc: Process pixel for scene with Dual-paraboloid shadow map
//-----------------------------------------------------------------------------
float4 PSPointDSM(float2 iTexCoord0 : TEXCOORD0,	// input texture coordinate for diffuse map
	float3 iNormal : TEXCOORD1,				// input the vertex normal in light view space
	float3 iVertexPos : TEXCOORD2,			// input the vertex position in light view space
	float4 iSMTexCoord : TEXCOORD3) : COLOR	// Shadow map texture coordinate(xy: front side, zw:back side)
{
	// Determine the distance from the light to the vertex and the direction
	float3 vLightDir = -iVertexPos;
	float fDistance = length(vLightDir);
	vLightDir = vLightDir / fDistance;
	float fNdotL = dot(vLightDir, iNormal);
	// Compute the per-pixel distance based attenuation
	float fAttenuation = clamp( 0, 1, 1 / (g_vLightAttenuation.x +
		g_vLightAttenuation.y * fDistance + g_vLightAttenuation.z * fDistance * fDistance) );
		
	float shadowDepth;
	fDistance -= 1.0f; // shadow bias
	if (iVertexPos.z >= 0)
		shadowDepth = tex2D(g_samShadowFront, iSMTexCoord.xy);
	else
		shadowDepth = tex2D(g_samShadowBack, iSMTexCoord.zw);
	float shadow = (fDistance / g_vLightAttenuation.w) <= shadowDepth;
	
	// Texture color * Lighting(Diffuse * Attenuation) Illumination
	return tex2D(g_samDiffuse, iTexCoord0) * ((fNdotL * g_vLightDiffuse) * fAttenuation * shadow);
}

//-----------------------------------------------------------------------------
// Pixel Shader: PSPointDSMHardware
// Desc: Process pixel for scene with Dual-paraboloid shadow map (Hardware Shadow Map)
//-----------------------------------------------------------------------------
float4 PSPointDSMHardware(float2 iTexCoord0 : TEXCOORD0,	// input texture coordinate for diffuse map
	float3 iNormal : TEXCOORD1,				// input the vertex normal in light view space
	float3 iVertexPos : TEXCOORD2,			// input the vertex position in light view space
	float4 iSMTexCoord : TEXCOORD3) : COLOR	// Shadow map texture coordinate(xy: front side, zw:back side)
{
	// Determine the distance from the light to the vertex and the direction
	float3 vLightDir = -iVertexPos;
	float fDistance = length(vLightDir);
	vLightDir = vLightDir / fDistance;
	float fNdotL = dot(vLightDir, iNormal);
	// Compute the per-pixel distance based attenuation
	float fAttenuation = clamp( 0, 1, 1 / (g_vLightAttenuation.x +
		g_vLightAttenuation.y * fDistance + g_vLightAttenuation.z * fDistance * fDistance) );
		
	float shadow;
	fDistance -= 1.0f;	// shadow bias
	if (iVertexPos.z >= 0)
		shadow = tex2Dproj(g_samShadowFront, float4(iSMTexCoord.xy, fDistance / g_vLightAttenuation.w, 1.0f));
	else
		shadow = tex2Dproj(g_samShadowBack, float4(iSMTexCoord.zw, fDistance / g_vLightAttenuation.w, 1.0f));
	
	// Texture color * Lighting(Diffuse * Attenuation) Illumination
	return tex2D(g_samDiffuse, iTexCoord0) * ((fNdotL * g_vLightDiffuse) * fAttenuation * shadow);
}

//-----------------------------------------------------------------------------
// Vertex Shader: VSPointCubeSM
// Desc: Process vertex for scnen with Cube shadow map
//-----------------------------------------------------------------------------
void VSPointCubeSM(float4 iPos : POSITION,	// input vertex position in object space
	float3 iNormal   : NORMAL,				// input vertex normal
	float2 iTexCoord : TEXCOORD0,			// input texture coordinate
	out float4 oPos : POSITION,			// output vertex position in clip space
	out float2 oTexCoord0 : TEXCOORD0,	// output texture coordinate for diffuse map
	out float3 oNormal : TEXCOORD1,		// output the vertex normal in light view space
	out float3 oVertexPos :TEXCOORD2)	// output the vertex position in light view space
{
	// copy texture coordinates
	oTexCoord0 = iTexCoord;

	// transform position to clip space
	oPos = mul(iPos, g_mWorldViewProj);
			
	// Transform unit vertex position from object space to light view space
	oVertexPos.xyz = mul(iPos, g_mWorldLightView);
	
	oNormal = mul(iNormal, g_mWorldLightView);	
}

//-----------------------------------------------------------------------------
// Pixel Shader: PSPointCubeSM
// Desc: Process pixel for scene with cube map
//-----------------------------------------------------------------------------
float4 PSPointCubeSM(float2 iTexCoord0 : TEXCOORD0,	// input texture coordinate for diffuse map
	float3 iNormal : TEXCOORD1,				// input the vertex normal in light view space
	float3 iVertexPos : TEXCOORD2) : COLOR	// input the vertex position in light view space
{
	// Determine the distance from the light to the vertex and the direction
	float3 vLightDir = -iVertexPos;
	float fDistance = length(vLightDir);
	vLightDir = vLightDir / fDistance;
	float fNdotL = dot(vLightDir, iNormal);
	// Compute the per-pixel distance based attenuation
	float fAttenuation = clamp( 0, 1, 1 / (g_vLightAttenuation.x +
		g_vLightAttenuation.y * fDistance + g_vLightAttenuation.z * fDistance * fDistance) );
		
	float shadow = (dot(iVertexPos, iVertexPos) / g_fLightRangeSquare) <= texCUBE(g_samCubeShadow, iVertexPos);
		
	// Texture color * Lighting(Diffuse * Attenuation) Illumination
	return tex2D(g_samDiffuse, iTexCoord0) * ((fNdotL * g_vLightDiffuse) * fAttenuation * shadow);
}

//-----------------------------------------------------------------------------
// Vertex Shader: VertLight
// Desc: Process vertex for the light object
//-----------------------------------------------------------------------------
void VertLight( float4 iPos : POSITION,
                float3 iNormal : NORMAL,
                float2 iTex : TEXCOORD0,
                out float4 oPos : POSITION,
                out float2 Tex : TEXCOORD0 )
{
	// transform position to clip space
    oPos = mul( iPos, g_mWorldViewProj );
    
    // Propagate texture coord
    Tex = iTex;
}

//-----------------------------------------------------------------------------
// Pixel Shader: PixLight
// Desc: Process pixel for the light object
//-----------------------------------------------------------------------------
float4 PixLight( float2 Tex : TEXCOORD0,
                 float4 vPos : TEXCOORD1 ) : COLOR
{
    return tex2D( g_samDiffuse, Tex );
}

//-----------------------------------------------------------------------------
// Vertex Shader: VSGenShadowMap
// Desc: Process vertex for the shadow map
//-----------------------------------------------------------------------------
void VSGenShadowMap(float4 iPos : POSITION,	// input vertex position in object space
	out float4 oPos : POSITION,
	out float2 oTexCoord0 : TEXCOORD0)		// output vertex position in clip space
{
	// transform position to clip space
	oPos = mul(iPos, g_mWorldViewProj);
	
	oTexCoord0 = oPos.zw;
}

//-----------------------------------------------------------------------------
// Pixel Shader: PSGenShadowMap
// Desc: Process pixel for the shadow map
//-----------------------------------------------------------------------------
float4 PSGenShadowMap(float2 iTexCoord0 : TEXCOORD0) : COLOR
{	
	// Depth is z / w
	return iTexCoord0.x / iTexCoord0.y;
}

//-----------------------------------------------------------------------------
// Vertex Shader: VSGenDistanceHardwareShadow
// Desc: Process vertex for the hardware shadow map with distance square
//-----------------------------------------------------------------------------
void VSGenDistanceHardwareShadow(float4 iPos : POSITION,	// input vertex position in object space
	out float4 oPos : POSITION,
	out float3 oVertexPos : TEXCOORD0)	// output the vertex position in light view space
{
	// transform position to clip space
	oPos = mul(iPos, g_mWorldViewProj);
	
	// Transform vertex position from object space to light view space
	float3 vVertexPos = mul(iPos, g_mWorldLightView).xyz;
	oPos.z = dot(vVertexPos, vVertexPos) * oPos.w / g_fLightRangeSquare;
}

//-----------------------------------------------------------------------------
// Vertex Shader: VSGenFrontDSM
// Desc: Process vertex for the front side Dual-paraboloid shadow map
//-----------------------------------------------------------------------------
void VSGenFrontDSM(float4 iPos : POSITION,	// input vertex position in object space
	out float4 oPos : POSITION,			// output vertex position in clip space
	out float4 vertexPos : TEXCOORD0)
{
	// Transform into light view space
	vertexPos.xyz = mul(iPos, g_mWorldLightView);
   
	oPos.z = length(vertexPos.xyz);
	vertexPos.xyz = vertexPos.xyz / oPos.z;
	oPos.x = vertexPos.x / (1 + vertexPos.z); // front
	oPos.y = vertexPos.y / (1 + vertexPos.z);
	vertexPos.w = oPos.z = (oPos.z) / g_vLightAttenuation.w;
	oPos.w = 1;
}

//-----------------------------------------------------------------------------
// Pixel Shader: PSGenFrontDSM
// Desc: Process pixel for the front side Dual-paraboloid shadow map
//-----------------------------------------------------------------------------
PS_OUTPUT PSGenFrontDSM(float4 iVertexPos : TEXCOORD0)
{	
	PS_OUTPUT Out;
	// Output the lit color
	if (iVertexPos.z < 0)
	{
		Out.Color = float4(1, 1, 1, 0.0);
		Out.Depth = -1;
	}
	else
	{
		Out.Color = float4(iVertexPos.www, 1);
		Out.Depth = iVertexPos.w;
	}
	return Out;
}

//-----------------------------------------------------------------------------
// Vertex Shader: VSGenBackDSM
// Desc: Process vertex for the back side Dual-paraboloid shadow map
//-----------------------------------------------------------------------------
void VSGenBackDSM(float4 iPos : POSITION,	// input vertex position in object space
	out float4 oPos : POSITION,			// output vertex position in clip space
	out float4 vertexPos : TEXCOORD0)
{
	// Transform into light view space
	vertexPos.xyz = mul(iPos, g_mWorldLightView);
   
	oPos.z = length(vertexPos.xyz);
	vertexPos.xyz = vertexPos.xyz / oPos.z;
	oPos.x = vertexPos.x / (1 - vertexPos.z); // back
	oPos.y = vertexPos.y / (1 - vertexPos.z);
	vertexPos.w = oPos.z = (oPos.z) / g_vLightAttenuation.w;
	oPos.w = 1;
}

//-----------------------------------------------------------------------------
// Pixel Shader: PSGenBackDSM
// Desc: Process pixel for the back side Dual-paraboloid shadow map
//-----------------------------------------------------------------------------
PS_OUTPUT PSGenBackDSM(float4 iVertexPos : TEXCOORD0)
{	
	PS_OUTPUT Out;
	// Output the lit color
	if (iVertexPos.z > 0)
	{
		Out.Color = float4(1, 1, 1, 0.0);
		Out.Depth = -1;
	}
	else
	{
		Out.Color = float4(iVertexPos.www, 1);
		Out.Depth = iVertexPos.w;
	}
	return Out;
}

//-----------------------------------------------------------------------------
// Vertex Shader: VSGenCubeShadowMap
// Desc: Process vertex for the cube shadow map
//-----------------------------------------------------------------------------
void VSGenCubeShadowMap(float4 iPos : POSITION,	// input vertex position in object space
	out float4 oPos : POSITION,
	out float3 oVertexPos : TEXCOORD0)	// output the vertex position in light view space
{
	// transform position to clip space
	oPos = mul(iPos, g_mWorldViewProj);
	
	// Transform vertex position from object space to light view space
	oVertexPos.xyz = mul(iPos, g_mWorldLightView);
}

//-----------------------------------------------------------------------------
// Pixel Shader: PSGenCubeShadowMap
// Desc: Process pixel for the cube shadow map
//-----------------------------------------------------------------------------
float4 PSGenCubeShadowMap(float3 iVertexPos : TEXCOORD0) : COLOR // output the vertex position in light view space
{
	return dot(iVertexPos, iVertexPos) / g_fLightRangeSquare;
}


//-----------------------------------------------------------------------------
// Technique: PointTSM
// Desc: Renders scene with Tetrahedron shadow map
//-----------------------------------------------------------------------------
technique PointTSM
{
	pass p0
	{		
		VertexShader = compile vs_2_0 VSPointTSM();
		PixelShader = compile ps_2_0 PSPointTSM();		
	}
}

//-----------------------------------------------------------------------------
// Technique: PointTSMHardware
// Desc: Renders scene with Tetrahedron shadow map (hardware shadow map)
//-----------------------------------------------------------------------------
technique PointTSMHardware
{
	pass p0
	{		
		VertexShader = compile vs_2_0 VSPointTSM();
		PixelShader = compile ps_2_0 PSPointTSMHardware();		
	}
}

//-----------------------------------------------------------------------------
// Technique: PointTSMLook
// Desc: Renders scene with Tetrahedron shadow map and lookup texture
//-----------------------------------------------------------------------------
technique PointTSMLook
{
	pass p0
	{		
		VertexShader = compile vs_2_0 VSPointTSMLook();
		PixelShader = compile ps_2_0 PSPointTSMLook();		
	}
}

//-----------------------------------------------------------------------------
// Technique: PointTSMLookHardware
// Desc: Renders scene with Tetrahedron shadow map (Hardware shadow map) and lookup texture
//-----------------------------------------------------------------------------
technique PointTSMLookHardware
{
	pass p0
	{		
		VertexShader = compile vs_2_0 VSPointTSMLook();
		PixelShader = compile ps_2_0 PSPointTSMLookHardware();		
	}
}

//-----------------------------------------------------------------------------
// Technique: PointDSM
// Desc: Renders scene with Dual-paraboloid shadow map
//-----------------------------------------------------------------------------
technique PointDSM
{
	pass p0
	{		
		VertexShader = compile vs_2_0 VSPointDSM();
		PixelShader = compile ps_2_0 PSPointDSM();		
	}
}

//-----------------------------------------------------------------------------
// Technique: PointDSMHardware
// Desc: Renders scene with Dual-paraboloid shadow map(Hardware Shadow Map)
//-----------------------------------------------------------------------------
technique PointDSMHardware
{
	pass p0
	{		
		VertexShader = compile vs_2_0 VSPointDSM();
		PixelShader = compile ps_2_0 PSPointDSMHardware();		
	}
}

//-----------------------------------------------------------------------------
// Technique: PointCubeSM
// Desc: Renders scene with cube shadow map
//-----------------------------------------------------------------------------
technique PointCubeSM
{
	pass p0
	{		
		VertexShader = compile vs_2_0 VSPointCubeSM();
		PixelShader = compile ps_2_0 PSPointCubeSM();		
	}
}


//-----------------------------------------------------------------------------
// Technique: RenderLight
// Desc: Renders the light object
//-----------------------------------------------------------------------------
technique RenderLight
{
    pass p0
    {
        VertexShader = compile vs_2_0 VertLight();
        PixelShader = compile ps_2_0 PixLight();
    }
}

//-----------------------------------------------------------------------------
// Technique: RenderShadow
// Desc: Renders the shadow map
//-----------------------------------------------------------------------------
technique RenderShadow
{
    pass p0
    {
		AlphaBlendEnable	= False;
        VertexShader = compile vs_2_0 VSGenShadowMap();
        PixelShader = compile ps_2_0 PSGenShadowMap();
    }
}

//-----------------------------------------------------------------------------
// Technique: RenderHardwareShadow
// Desc: Renders the hardware shadow map
//-----------------------------------------------------------------------------
technique RenderHardwareShadow
{
    pass p0
    {
		AlphaBlendEnable	= False;
		ColorWriteEnable = 0;     // no need to render to color, we only need z
        VertexShader = compile vs_2_0 VSGenShadowMap();
    }
}

//-----------------------------------------------------------------------------
// Technique: RenderHardwareShadow
// Desc: Renders the hardware shadow map with distance square
//-----------------------------------------------------------------------------
technique RenderDistanceHardwareShadow
{
    pass p0
    {
		AlphaBlendEnable	= False;
		ColorWriteEnable = 0;     // no need to render to color, we only need z
        VertexShader = compile vs_2_0 VSGenDistanceHardwareShadow();
    }
}

//-----------------------------------------------------------------------------
// Technique: RenderFrontShadow
// Desc: Renders the front side of shadow map
//-----------------------------------------------------------------------------
technique RenderFrontShadow
{
    pass p0
    {
		AlphaBlendEnable	= False;
        VertexShader = compile vs_2_0 VSGenFrontDSM();
        PixelShader = compile ps_2_0 PSGenFrontDSM();
    }
}

//-----------------------------------------------------------------------------
// Technique: RenderBackShadow
// Desc: Renders the back side of shadow map
//-----------------------------------------------------------------------------
technique RenderBackShadow
{
    pass p0
    {
		AlphaBlendEnable	= False;
        VertexShader = compile vs_2_0 VSGenBackDSM();
        PixelShader = compile ps_2_0 PSGenBackDSM();
    }
}

//-----------------------------------------------------------------------------
// Technique: RenderFrontHardwareShadow
// Desc: Renders the front side of hardware shadow map
//-----------------------------------------------------------------------------
technique RenderFrontHardwareShadow
{
    pass p0
    {
		AlphaBlendEnable	= False;
		ColorWriteEnable = 0;     // no need to render to color, we only need z
        VertexShader = compile vs_2_0 VSGenFrontDSM();
        PixelShader = compile ps_2_0 PSGenFrontDSM();
    }
}

//-----------------------------------------------------------------------------
// Technique: RenderBackHardwareShadow
// Desc: Renders the back side of hardware shadow map
//-----------------------------------------------------------------------------
technique RenderBackHardwareShadow
{
    pass p0
    {
		AlphaBlendEnable	= False;
		ColorWriteEnable = 0;     // no need to render to color, we only need z
        VertexShader = compile vs_2_0 VSGenBackDSM();
        PixelShader = compile ps_2_0 PSGenBackDSM();
    }
}

//-----------------------------------------------------------------------------
// Technique: RenderCubeShadow
// Desc: Renders the shadow map
//-----------------------------------------------------------------------------
technique RenderCubeShadow
{
    pass p0
    {
		AlphaBlendEnable	= False;
        VertexShader = compile vs_2_0 VSGenCubeShadowMap();
        PixelShader = compile ps_2_0 PSGenCubeShadowMap();
    }
}