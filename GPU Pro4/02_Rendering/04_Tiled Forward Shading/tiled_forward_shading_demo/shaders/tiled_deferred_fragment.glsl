#version 330 compatibility

#define UBERSHADER 1 // 0 //1


#include "globals.glsl"
#include "srgb.glsl"

#include "LightAccel_Tiled.glsl"
#include "ShadingModel.glsl"

#if NUM_MSAA_SAMPLES == 1
uniform sampler2D diffuseTex;
uniform sampler2D specularShininessTex;
uniform sampler2D ambientTex;
uniform sampler2D normalTex;
uniform sampler2D depthTex;
#else // NUM_MSAA_SAMPLES != 1
uniform sampler2DMS diffuseTex;
uniform sampler2DMS specularShininessTex;
uniform sampler2DMS ambientTex;
uniform sampler2DMS normalTex;
uniform sampler2DMS depthTex;
#endif // ~ NUM_MSAA_SAMPLES

out vec4 resultColor;

vec3 unProject(vec2 fragmentPos, float fragmentDepth)
{
	vec4 pt = inverseProjectionMatrix * vec4(fragmentPos.x * 2.0 - 1.0, fragmentPos.y * 2.0 - 1.0, 2.0 * fragmentDepth - 1.0, 1.0);
	return vec3(pt.x, pt.y, pt.z) / pt.w;
}

vec3 fetchPosition(vec2 p, int sample)
{
	vec2 fragmentPos = vec2(p.x * invFbSize.x, p.y * invFbSize.y);
	float d = texelFetch(depthTex, ivec2(p), sample).x;
	return unProject(fragmentPos, d);
}

void main()
{
	ivec2 sampleCoord = ivec2(gl_FragCoord.xy);

	vec3 shading = vec3(0);

#	if 0 // XXX-workaround for crash
	
	/* So, for some reason the following loop crashes and burns on various
	   NVIDIA drivers (or triggers some error during linking). (Linux = 
	   `Segmentation Fault (core dumpled)' in some -glcore.so.
		 Windows: Access violation reading location 0x0000000000000000 >	nvoglv64.dll!0000000063681341()).

	   WARNING: the Ubershader/material stuff is not implemented in the loop below.
	*/

	for( int sampleIndex = 0; sampleIndex < NUM_MSAA_SAMPLES; ++sampleIndex )
	{
		// Fetch stuff
		SampleData data;
		data.position = fetchPosition( gl_FragCoord.xy, sampleIndex );
		data.normal = texelFetch( normalTex, sampleCoord, sampleIndex ).xyz;

		vec4 specShi = texelFetch( specularShininessTex, sampleCoord, sampleIndex );
		data.diffuse = texelFetch( diffuseTex, sampleCoord, sampleIndex ).xyz;
		data.ambient = texelFetch( ambientTex, sampleCoord, sampleIndex ).xyz;
		data.specular = specShi.xyz;

		data.alpha = 1.0;
		data.specularExponent = specShi.w;

		// Shade
		float dummy;
		shading += SHADING_MODEL_INTERFACE(SHADING_MODEL,shade)( data, dummy );
	}
#	else
	/* Workaround = manually unroll the loop for great glory.
	 * As this works on AMD cards, this is the default.
	 */

#if UBERSHADER
#		define SHADE_LOOP_BODY(sampleIndex) \
		if( sampleIndex < NUM_MSAA_SAMPLES ) { \
			SampleData data; \
			data.position = fetchPosition( gl_FragCoord.xy, sampleIndex ); \
			data.normal = texelFetch( normalTex, sampleCoord, sampleIndex ).xyz; \
			\
			vec4 specShi = texelFetch(specularShininessTex, sampleCoord, sampleIndex); \
			vec4 diffuseShadingMod = texelFetch(diffuseTex, sampleCoord, sampleIndex);\
			\
			int shadingModel = int(diffuseShadingMod.w); \
			data.diffuse = diffuseShadingMod.xyz; \
			data.ambient = texelFetch( ambientTex, sampleCoord, sampleIndex ).xyz; \
			data.specular = specShi.xyz; \
			\
			data.alpha = 1.0; \
			data.specularExponent = specShi.w; \
			\
			float dummy; \
			vec3 result = vec3(0.0);\
			SHADING_MODEL_INTERFACE_RT( shadingModel, shade, result, (data,dummy) ); \
			shading += result; \
		} \
		/*ENDM*/
#else
#		define SHADE_LOOP_BODY(sampleIndex) \
		if( sampleIndex < NUM_MSAA_SAMPLES ) { \
			SampleData data; \
			data.position = fetchPosition( gl_FragCoord.xy, sampleIndex ); \
			data.normal = texelFetch( normalTex, sampleCoord, sampleIndex ).xyz; \
			\
			vec4 specShi = texelFetch(specularShininessTex, sampleCoord, sampleIndex); \
			vec4 diffuseShadingMod = texelFetch(diffuseTex, sampleCoord, sampleIndex);\
			\
			int shadingModel = int(diffuseShadingMod.w); \
			data.diffuse = diffuseShadingMod.xyz; \
			data.ambient = texelFetch( ambientTex, sampleCoord, sampleIndex ).xyz; \
			data.specular = specShi.xyz; \
			\
			data.alpha = 1.0; \
			data.specularExponent = specShi.w; \
			\
			float dummy; \
			vec3 result = vec3(0.0);\
			result = SHADING_MODEL_INTERFACE(SHADING_MODEL_DEFAULT,shade)(data,dummy); \
			shading += result; \
		} \
		/*ENDM*/
#endif

		SHADE_LOOP_BODY(0); 
		SHADE_LOOP_BODY(1); 
		SHADE_LOOP_BODY(2);
		SHADE_LOOP_BODY(3); 
		SHADE_LOOP_BODY(4); 
		SHADE_LOOP_BODY(5);
		SHADE_LOOP_BODY(6); 
		SHADE_LOOP_BODY(7); 
		SHADE_LOOP_BODY(8);
		SHADE_LOOP_BODY(9); 
		SHADE_LOOP_BODY(10); 
		SHADE_LOOP_BODY(11);
		SHADE_LOOP_BODY(12); 
		SHADE_LOOP_BODY(13); 
		SHADE_LOOP_BODY(14);
		SHADE_LOOP_BODY(15); 

#		undef SHADE_LOOP_BODY
#	endif 

	resultColor = vec4( toSrgb(shading.xyz/NUM_MSAA_SAMPLES), 1.0 );
}
