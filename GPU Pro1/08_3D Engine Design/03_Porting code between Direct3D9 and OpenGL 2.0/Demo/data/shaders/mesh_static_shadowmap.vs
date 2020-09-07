float4x4 worldTransform;
float4x4 viewProjTransform;



struct VS_INPUT
{
	float4 position: POSITION;
	float2 texCoord0: TEXCOORD2;
};



struct VS_OUTPUT
{
	float4 position: POSITION;
	float2 texCoord0: TEXCOORD0;
};



VS_OUTPUT main(VS_INPUT input)
{
	VS_OUTPUT output;

	// world-space
	float4 transformedPosition = mul(input.position, worldTransform);

	output.position = mul(transformedPosition, viewProjTransform);
	output.texCoord0 = input.texCoord0;

	return output;
}
