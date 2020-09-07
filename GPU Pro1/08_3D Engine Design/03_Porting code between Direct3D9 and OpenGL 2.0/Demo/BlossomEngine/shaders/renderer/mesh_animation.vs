/* $Id: mesh_animation.vs 126 2009-08-22 17:08:39Z maxest $ */



float4x4 worldTransform;
float4x4 worldTransformInversed;
float4x4 animationMatrices[20];
float4x4 viewProjTransform;



struct VS_INPUT
{
	float4 position: POSITION;
	float3 normal: NORMAL;
	float3 tangent: TEXCOORD0;
	float3 bitangent: TEXCOORD1;
	float2 texCoord0: TEXCOORD2;
	float nodeIndex: TEXCOORD3;
};



struct VS_OUTPUT
{
	float4 position: POSITION;
	float3 normal: TEXCOORD0;
	float2 texCoord0: TEXCOORD1;
	float3 lightDirection: TEXCOORD2; // -pixelToLight
};



VS_OUTPUT main(VS_INPUT input)
{
	VS_OUTPUT output;

	float3 lightDirection = normalize(float3(-1.0, -1.0, -1.0));

	// world-space
	float4 transformedPosition = mul(mul(input.position, animationMatrices[(int)input.nodeIndex]), worldTransform);

	// object-space
	input.normal = mul(input.normal, (float3x3)animationMatrices[(int)input.nodeIndex]);
	input.tangent = mul(input.tangent, (float3x3)animationMatrices[(int)input.nodeIndex]);
	input.bitangent = mul(input.bitangent, (float3x3)animationMatrices[(int)input.nodeIndex]);
	output.lightDirection = mul(lightDirection, (float3x3)worldTransformInversed);

	float3x3 fromObjectToTangentSpaceTransform;
	fromObjectToTangentSpaceTransform[0] = input.tangent;
	fromObjectToTangentSpaceTransform[1] = input.bitangent;
	fromObjectToTangentSpaceTransform[2] = input.normal;
	fromObjectToTangentSpaceTransform = transpose(fromObjectToTangentSpaceTransform);

	// tangent-space
	output.normal = mul(input.normal, fromObjectToTangentSpaceTransform);
	output.lightDirection = mul(output.lightDirection, fromObjectToTangentSpaceTransform);

	output.position = mul(transformedPosition, viewProjTransform);
	output.texCoord0 = input.texCoord0;

	return output;
}
