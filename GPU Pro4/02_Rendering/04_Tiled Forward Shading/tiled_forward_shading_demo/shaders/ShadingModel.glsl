#ifndef SHADINGMODEL_GLSL_DF32A9A9_D177_4C0F_A6E2_D6197DBE4B31
#define SHADINGMODEL_GLSL_DF32A9A9_D177_4C0F_A6E2_D6197DBE4B31

// Conf
#define ENABLE_TANGENT_AND_BITANGENT 1

// Shading Models
#define SHADING_MODEL_NONE 0
#define SHADING_MODEL_DEFAULT 1
#define SHADING_MODEL_RIMLIGHT 2
#define SHADING_MODEL_CARPAINT 3

#if !defined(SHADING_MODEL)
#	define SHADING_MODEL SHADING_MODEL_NONE
#endif // ~ !defined(SHADING_MODEL)

#if !defined(SHADING_MODEL_CONSTANTS_ONLY)
// Sample Data
struct SampleData
{
	vec3 position;
	vec3 normal;

	vec3 ambient; // Note: ambient is a constant term, and could include emissive
	vec3 diffuse;
	vec3 specular;

	float alpha;
	float specularExponent;

#	if ENABLE_TANGENT_AND_BITANGENT
	vec3 tangent;
	vec3 bitangent;
#	endif // ~ ENABLE_TANGENT_AND_BITANGENT
};

// Helpers
#define SHADING_MODEL_INTERFACE_(model, kind) _sm_##model##_##kind
#define SHADING_MODEL_INTERFACE(model, kind) SHADING_MODEL_INTERFACE_(model,kind)

// Include implementations
#if UBERSHADER || SHADING_MODEL == SHADING_MODEL_DEFAULT
	#include "shadingModels/ShadingModel_Default.glsl"
#endif

#if UBERSHADER || SHADING_MODEL == SHADING_MODEL_RIMLIGHT
	#include "shadingModels/ShadingModel_RimLight.glsl"
#endif

#if UBERSHADER || SHADING_MODEL == SHADING_MODEL_CARPAINT
	#include "shadingModels/ShadingModel_Carpaint.glsl"
#endif


// Helpers
#define SHADING_MODEL_INTERFACE_RT_(rt, model, kind, result, params) \
	if( rt == model ) { result = SHADING_MODEL_INTERFACE(model,kind) params; } \
	/*END*/

#define SHADING_MODEL_INTERFACE_RT(rt, kind, result, params) \
	SHADING_MODEL_INTERFACE_RT_(rt, SHADING_MODEL_DEFAULT, kind, result, params ) \
	else SHADING_MODEL_INTERFACE_RT_(rt, SHADING_MODEL_RIMLIGHT, kind, result, params ) \
	else SHADING_MODEL_INTERFACE_RT_(rt, SHADING_MODEL_CARPAINT, kind, result, params ) \
	else { result = vec3(1.0,0.0,0.0); }
	/*END*/

#endif // ~ defined(SHADING_MODEL_CONSTANTS_ONLY)

#endif // SHADINGMODEL_GLSL_DF32A9A9_D177_4C0F_A6E2_D6197DBE4B31
