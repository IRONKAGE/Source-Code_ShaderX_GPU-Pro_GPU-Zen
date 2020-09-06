#ifndef SHADINGMODEL_CARPAINT_GLSL_A761B08B_9EEC_4C7C_B28F_43B04101E87B
#define SHADINGMODEL_CARPAINT_GLSL_A761B08B_9EEC_4C7C_B28F_43B04101E87B

//#  include "ObjModel.glsl"
#include "../globals.glsl"

#include "../Noise3D.glsl"
#include "../ColorSpaces.glsl"

vec3 SHADING_MODEL_INTERFACE(SHADING_MODEL_CARPAINT,shade)( in SampleData aSampleData, out float aResultAlpha )
{
	vec3 viewDir = -normalize(aSampleData.position);

	// micro flakes
	vec4 wp = inverse(viewMatrix) * vec4(aSampleData.position,1.0);
	wp /= wp.w;

	vec3 wp0 = wp.xyz * 666.0;
	vec3 flake0 = vec3( snoise(wp0.xyz), snoise(wp0.yzx), snoise(wp0.zxy) );

	vec3 wp1 = wp0 * 2.0;
	vec3 flake1 = vec3( snoise(wp1.xyz), snoise(wp1.yzx), snoise(wp1.zxy) );

	float ll = length(aSampleData.position);
	float f1 = smoothstep( 0.2, 0.7, ll );
	float f2 = smoothstep( 0.6, 1.5, ll );
	vec3 flake = mix( flake1, mix( flake0, vec3(0.5), f2 ), f1 );


	// invent some colors
	vec3 diffuseHSL = rgb_to_hsl( aSampleData.diffuse );
	diffuseHSL.g = min( 1.0, diffuseHSL.g+0.2 );
	vec3 diffuse = hsl_to_rgb( diffuseHSL );

	diffuseHSL.r = mod( diffuseHSL.r+0.1, 1.0 );
	vec3 paintColor = hsl_to_rgb( diffuseHSL );

	vec3 halfColor = mix( diffuse, paintColor, 0.4 );

	diffuseHSL.b = min( 1.0, diffuseHSL.b+0.3 );
	vec3 flakeColor = hsl_to_rgb( diffuseHSL );

	// weird stuff
	float flakePerturbA = 0.1;
	float normalPerturb = 1.0;
	float flakePerturb = 1.0;

	vec3 np1 = flakePerturbA * flake + normalPerturb * aSampleData.normal;
	vec3 np2 = flakePerturb * (0.2*flake+aSampleData.normal);

	float NdotV = max( 0.0, dot( aSampleData.normal, viewDir ) );
	
	float fres1 = max( 0.0, dot( normalize(np1), viewDir ) );
	float fres2 = max( 0.0, dot( normalize(np2), viewDir ) );

	float fres1Sq = fres1*fres1;
	vec3 paint = fres1 * diffuse
		+ fres1Sq * halfColor 
		+ fres1Sq*fres1Sq * paintColor
		+ pow(fres2, 16 ) * flakeColor;

	// lighting
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

		result += nDotL * attenuation * (paint + spec) * lightData.color;
	FOR_EACH_LIGHT_END()

	// store/return results
	aResultAlpha =  aSampleData.alpha + fresnelTerm * (1.0 - aSampleData.alpha);

	return result + 2.0*paint*ambientGlobal;
}

#endif // SHADINGMODEL_CARPAINT_GLSL_A761B08B_9EEC_4C7C_B28F_43B04101E87B
