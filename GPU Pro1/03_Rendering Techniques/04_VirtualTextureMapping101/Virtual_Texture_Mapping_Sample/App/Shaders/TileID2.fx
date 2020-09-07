cbuffer camera
{
	float4x4	WorldViewProjection : WorldViewProjection;
}

struct VS_OUTPUT
{
	float4	Position	: SV_POSITION;
	float2	UV			: TEXCOORD0;
};

cbuffer VTM_Engine_Settings
{
	float MaximumMipMapLevel;
	float MipMapScaleFactor;
}

cbuffer vtm
{
	float	TextureID;
	float2	TextureSize;
	float2	TileCount;
}

RasterizerState DisableCulling
{
    CullMode = NONE;
};

DepthStencilState DepthEnabling
{
	DepthEnable = TRUE;
};

BlendState DisableBlend
{
	BlendEnable[0] = FALSE;
};

VS_OUTPUT std_VS  (
	float4	Pos : POSITION,
	float2	UV	: TEXCOORD0)
{
	VS_OUTPUT OUT = (VS_OUTPUT)0;
	
	OUT.Position =  mul(Pos, WorldViewProjection);
	OUT.UV = UV;
	
	return OUT;
}

#include "mipLevel.shi"

float4 std_PS(VS_OUTPUT In) : SV_Target
{
	float mipLevel = ComputeMipMapLevel (In.UV, TextureSize, MipMapScaleFactor);
	
	float4 result;
	
	float rIndex 		= floor (min (mipLevel, MaximumMipMapLevel));
	float2 pageID 		= floor (In.UV * TileCount / exp2(rIndex));
	result.rg 			= pageID;
	result.b 			= rIndex;
	
	result.a 			= TextureID;
	
	return result / 255.0;
}

///// TECHNIQUES /////////////////////////////


technique10 Render 
{
    pass p0
    {
        SetVertexShader( CompileShader( vs_4_0, std_VS() ) );
        SetGeometryShader( NULL );
        SetRasterizerState(DisableCulling);     
        SetPixelShader( CompileShader( ps_4_0, std_PS() ) );  
    }
}

/////////////////////////////////////// eof //
