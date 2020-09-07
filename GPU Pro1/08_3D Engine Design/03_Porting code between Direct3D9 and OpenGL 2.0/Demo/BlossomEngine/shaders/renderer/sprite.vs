/* $Id: sprite.vs 180 2009-08-27 15:32:36Z maxest $ */



float4x4 worldTransform;
float4x4 viewProjTransform;



struct VS_INPUT
{
	float4 position: POSITION;
	float4 color: COLOR;
	float2 texCoord0: TEXCOORD0;
	float3 transformRow0: TEXCOORD1;
	float3 transformRow1: TEXCOORD2;
	float3 transformRow2: TEXCOORD3;
	float3 transformRow3: TEXCOORD4;
};



struct VS_OUTPUT
{
	float4 position: POSITION;
	float4 color: COLOR;
	float2 texCoord0: TEXCOORD0;
};



VS_OUTPUT main(VS_INPUT input)
{
	VS_OUTPUT output;

	float4x4 spriteTransform = { float4(input.transformRow0, 0.0),
								 float4(input.transformRow1, 0.0),
								 float4(input.transformRow2, 0.0), 
								 float4(input.transformRow3, 1.0) };
	float4 transformedPosition = mul(mul(input.position, spriteTransform), worldTransform);

	output.position = mul(transformedPosition, viewProjTransform);
	output.color = input.color;
	output.texCoord0 = input.texCoord0;

	return output;
}
