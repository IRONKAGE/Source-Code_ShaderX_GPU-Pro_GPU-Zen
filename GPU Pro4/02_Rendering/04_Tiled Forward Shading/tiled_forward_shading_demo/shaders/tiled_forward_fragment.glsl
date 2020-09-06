#version 330 compatibility

#define UBERSHADER 0


#include "globals.glsl"
#include "srgb.glsl"
#include "ObjModel.glsl"

#include "LightAccel_Tiled.glsl"
#include "ShadingModel.glsl"

in vec3 v2f_normal;
in vec3	v2f_tangent;
in vec3	v2f_bitangent;
in vec3	v2f_position;
in vec2 v2f_texCoord;

out vec4 resultColor;

void main()
{
#	if ENABLE_ALPHA_TEST
	// Alpha test
	if (texture2D(opacity_texture, v2f_texCoord).r < 0.5)
		discard;
#	endif // ~ ENABLE_ALPHA_TEST

	// compute per-sample information
	vec3 position = v2f_position;
	vec3 viewDir = -normalize(position);

#	if ENABLE_TANGENT_AND_BITANGENT
	vec3 tangent, bitangent;
#	endif

	vec3 normal = normalize(v2f_normal); 
	{
		vec3 normalSpaceX = normalize(v2f_tangent);
		vec3 normalSpaceY = normalize(v2f_bitangent);
		vec3 normalSpaceZ = normalize(v2f_normal);

		vec3 normalMapSample = texture2D(normal_texture, v2f_texCoord).xyz * vec3(2.0) - vec3(1.0);

		normal = normalize(normalMapSample.x * normalSpaceX + normalMapSample.y * normalSpaceY + normalMapSample.z * normalSpaceZ);

#		if ENABLE_TANGENT_AND_BITANGENT
		// tangent and bitangent
		bitangent = normalize(cross(cross(normalSpaceY, normal), normal)); 
		tangent = normalize(cross( normal, bitangent ));
#		endif
	}


	vec3 diffuse = texture2D(diffuse_texture, v2f_texCoord).rgb * material_diffuse_color;
	vec3 specular = texture2D(specular_texture, v2f_texCoord).rgb * material_specular_color;

	// Do shading: gather Sample specific data into SampleData and pass on to the
	SampleData data;
	data.position = position;
	data.normal = normal;

	data.diffuse = diffuse;
	data.ambient = diffuse * ambientGlobal;
	data.specular = specular;

	data.alpha = material_alpha;
	data.specularExponent = material_specular_exponent;

#	if ENABLE_TANGENT_AND_BITANGENT
	data.tangent = tangent;
	data.bitangent = bitangent;
#	endif

	float resultAlpha;
	vec3 shading = SHADING_MODEL_INTERFACE(SHADING_MODEL,shade)( data, resultAlpha );

	// Store result
	resultColor = vec4( toSrgb(shading.xyz), resultAlpha );
}
