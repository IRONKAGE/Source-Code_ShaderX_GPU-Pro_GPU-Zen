#ifndef LIGHTACCEL_TILED_GLSL_AD31E870_CDA2_4CA4_AD39_FF0ACC0306A9
#define LIGHTACCEL_TILED_GLSL_AD31E870_CDA2_4CA4_AD39_FF0ACC0306A9

// Data for Tiled Structure
uniform isamplerBuffer tileLightIndexListsTex;

uniform LightGrid
{
	ivec4 lightGridCountOffsets[LIGHT_GRID_MAX_DIM_X * LIGHT_GRID_MAX_DIM_Y];
};

uniform LightColors
{
	vec4 g_light_color[NUM_POSSIBLE_LIGHTS];
};
uniform LightPositionsRanges
{
	vec4 g_light_position_range[NUM_POSSIBLE_LIGHTS];
};

// Common stuff
struct LightData
{
	vec3 position;
	float range;

	vec3 color;
};

// Accessors to Light Data
void light_get_data( in int aLightId, out LightData aLightData )
{
	aLightData.position = g_light_position_range[aLightId].xyz;
	aLightData.range = g_light_position_range[aLightId].w;

	aLightData.color = g_light_color[aLightId].xyz;
}

// FOR_EACH_LIGHT_BEGIN/END()
#define FOR_EACH_LIGHT_BEGIN( lightIdent, lightData ) \
	{ \
		ivec2 tileIndex = ivec2(int(gl_FragCoord.x) / LIGHT_GRID_TILE_DIM_X, int(gl_FragCoord.y) / LIGHT_GRID_TILE_DIM_Y); \
		ivec2 countOffset = lightGridCountOffsets[tileIndex.x + tileIndex.y * LIGHT_GRID_MAX_DIM_X].xy; \
		\
		LightData lightData;\
		for( int lightIndex = 0; lightIndex < countOffset.x; ++lightIndex )	\
		{ \
			int lightIdent = texelFetch( tileLightIndexListsTex, countOffset.y + lightIndex).x; \
			light_get_data(lightIdent, lightData);

#define FOR_EACH_LIGHT_END() \
		}	\
	}

#endif // LIGHTACCEL_TILED_GLSL_AD31E870_CDA2_4CA4_AD39_FF0ACC0306A9
