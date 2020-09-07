/* $Id: mesh_animation_interpolation.vs 126 2009-08-22 17:08:39Z maxest $ */



float4x4 worldTransform;
float4x4 worldTransformInversed;
float4x4 animationMatrices1[10];
float4x4 animationMatrices2[10];
float4x4 viewProjTransform;

float interpolationProgress;



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

	float3 lightDirection = float3(-1.0, -1.0, -1.0);

	float2 interpolationFactors = float2((1.0-interpolationProgress), interpolationProgress);

	// world-space
	float4 transformedPosition1 = mul(mul(input.position, animationMatrices1[(int)input.nodeIndex]), worldTransform);
	float4 transformedPosition2 = mul(mul(input.position, animationMatrices2[(int)input.nodeIndex]), worldTransform);
	float4 transformedPosition = transformedPosition1*interpolationFactors.x + transformedPosition2*interpolationFactors.y;

	// object-space
	float3 transformedNormal1 = mul(input.normal, (float3x3)animationMatrices1[(int)input.nodeIndex]);
	float3 transformedNormal2 = mul(input.normal, (float3x3)animationMatrices2[(int)input.nodeIndex]);
	float3 transformedTangent1 = mul(input.tangent, (float3x3)animationMatrices1[(int)input.nodeIndex]);
	float3 transformedTangent2 = mul(input.tangent, (float3x3)animationMatrices2[(int)input.nodeIndex]);
	float3 transformedBitangent1 = mul(input.bitangent, (float3x3)animationMatrices1[(int)input.nodeIndex]);
	float3 transformedBitangent2 = mul(input.bitangent, (float3x3)animationMatrices2[(int)input.nodeIndex]);
	input.normal = transformedNormal1*interpolationFactors.x + transformedNormal2*interpolationFactors.y;
	input.tangent = transformedTangent1*interpolationFactors.x + transformedTangent2*interpolationFactors.y;
	input.bitangent = transformedBitangent1*interpolationFactors.x + transformedBitangent2*interpolationFactors.y;
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
