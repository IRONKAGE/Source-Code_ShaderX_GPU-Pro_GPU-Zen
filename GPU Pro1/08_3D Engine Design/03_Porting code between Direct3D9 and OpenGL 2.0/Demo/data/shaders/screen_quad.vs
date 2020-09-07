struct VS_INPUT
{
	float4 position: POSITION;
	float2 texCoord0: TEXCOORD0;
};



struct VS_OUTPUT
{
	float4 position: POSITION;
	float2 texCoord0: TEXCOORD0;
};



VS_OUTPUT main(VS_INPUT input)
{
	VS_OUTPUT output;
	
	output.position = input.position;
	output.texCoord0 = input.texCoord0;

	return output;
}
