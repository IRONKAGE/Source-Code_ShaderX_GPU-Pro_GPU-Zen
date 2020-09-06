#ifndef SHADINGMODEL_DEFAULT_GLSL_13609257_E75A_4CB3_9B3A_B770BDC6B602
#define SHADINGMODEL_DEFAULT_GLSL_13609257_E75A_4CB3_9B3A_B770BDC6B602

//#  include "ObjModel.glsl"
#include "../LightAccel_Tiled.glsl"

vec3 SHADING_MODEL_INTERFACE(SHADING_MODEL_DEFAULT,shade)( in SampleData aSampleData, out float aResultAlpha )
{
	vec3 viewDir = -normalize(aSampleData.position);

	float fresnelTerm = pow(clamp(1.0 + dot(-viewDir, aSampleData.normal),0.0,1.0), 5.0);
	vec3 fresnelSpec = aSampleData.specular + fresnelTerm * (vec3(1.0)-aSampleData.specular);

	vec3 result = vec3(0.0);
	FOR_EACH_LIGHT_BEGIN( lightId, lightData )
		vec3 lightDir = lightData.position - aSampleData.position;
		float lightDist = length(lightDir);
		lightDir = normalize(lightDir);

		float inner = 0.0f;
		float nDotL = max( dot( aSampleData.normal, lightDir ), 0.0 );

		float attenuation = max( 1.0 - max( 0.0, (lightDist-inner) / (lightData.range-inner) ), 0.0 );
		float normFactor = (aSampleData.specularExponent + 2.0) / 8.0;

		vec3 h = normalize(lightDir + viewDir);
		vec3 spec = fresnelSpec * normFactor * pow( max( 0.0, dot( h, aSampleData.normal ) ), aSampleData.specularExponent );

		result += nDotL * attenuation * (aSampleData.diffuse + spec) * lightData.color;
	FOR_EACH_LIGHT_END()

	// store/return results
	aResultAlpha =  aSampleData.alpha + fresnelTerm * (1.0 - aSampleData.alpha);
	return result + aSampleData.ambient;
}

#endif // SHADINGMODEL_DEFAULT_GLSL_13609257_E75A_4CB3_9B3A_B770BDC6B602
