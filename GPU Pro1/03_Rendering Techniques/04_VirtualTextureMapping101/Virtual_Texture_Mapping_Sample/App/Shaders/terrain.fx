cbuffer camera
{
	float4x4	WorldViewProjection : WorldViewProjection;
}

struct VS_OUTPUT
{
	float4	Position	:	SV_POSITION;
	float2	TextureUV	:	TEXCOORD0;
	float3	Normal		:	NORMAL;
};


cbuffer PageCacheSettings
{
	float2		Cache_PageSize;
	float2		Cache_Size;
}

Texture2D	Cache_Texture;
Texture2D	Cache_Indirection;

cbuffer VirtualTextureSettings
{
	float2		Virtual_TextureSize;
	float		Virtual_MaxMipMapLevel;
}

VS_OUTPUT RenderSceneVS (
	float4	pos		: POSITION,
	float2	uv		: TEXCOORD0,
	float3	normal	: NORMAL
)
{
	VS_OUTPUT Output;
	
	Output.Position = mul (pos, WorldViewProjection);	
	Output.TextureUV = uv;
	Output.Normal = normal;
	
	return Output;
}

SamplerState TextureSampler
{
    Filter = ANISOTROPIC;
	MaxAnisotropy = 4;
    AddressU = Clamp;
    AddressV = Clamp;
};

SamplerState PointSampler
{
    Filter = MIN_MAG_POINT_MIP_LINEAR;
    AddressU = Clamp;
    AddressV = Clamp;
};

float4 RenderScenePS (VS_OUTPUT In) : SV_Target
{			
	float3 pageData = Cache_Indirection.Sample (PointSampler, In.TextureUV);
	float2 inPageOffset = frac(In.TextureUV * exp2(pageData.z));
	inPageOffset *= Cache_PageSize;
	
	float2 grad = Cache_PageSize * exp2(pageData.z);
	float4 grad_ddx_ddy = float4(ddx (In.TextureUV), ddy (In.TextureUV)) * grad.xyxy;
	
	// Use the following to see the gradient lengths and discontinuities
	// return float4 (length (ddx (pageData.xy + inPageOffset)), length (ddy (pageData.xy + inPageOffset)), 0, 1);
	
	// Use this to see page-cache UV coordinates
	// return float4 (pageData.xy + inPageOffset, 0, 1);
	
	float diffuse = dot (In.Normal, normalize (float3 (-0.45, 1, 0.45)));
	
	return pow(diffuse,4) * Cache_Texture.SampleGrad (TextureSampler, pageData.xy + inPageOffset, grad_ddx_ddy.xy, grad_ddx_ddy.zw);
}

RasterizerState rsFilled { FillMode = Solid; };

technique10 Render
{
	pass p0
	{
		SetVertexShader (CompileShader (vs_4_0, RenderSceneVS ()));
		SetGeometryShader (NULL);
		SetRasterizerState (rsFilled);
		SetPixelShader (CompileShader (ps_4_0, RenderScenePS ()));	
	}
}