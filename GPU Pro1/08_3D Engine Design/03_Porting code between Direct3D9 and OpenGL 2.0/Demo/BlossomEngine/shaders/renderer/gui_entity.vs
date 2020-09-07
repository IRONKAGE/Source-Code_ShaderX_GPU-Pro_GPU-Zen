/* $Id: gui_entity.vs 126 2009-08-22 17:08:39Z maxest $ */



struct VS_INPUT
{
	float2 position: POSITION;
	float4 color: COLOR;
	float2 texCoord0: TEXCOORD0;
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

	output.position = float4(input.position, 0.0, 1.0);
	output.color = input.color;
	output.texCoord0 = input.texCoord0;

	return output;
}
