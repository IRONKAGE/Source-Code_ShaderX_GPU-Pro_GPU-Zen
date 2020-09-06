#ifndef COLORSPACES_GLSL_34800F2E_2C0A_4035_B227_98599373E023
#define COLORSPACES_GLSL_34800F2E_2C0A_4035_B227_98599373E023

vec3 rgb_to_hsl( in vec3 aRGB );
vec3 hsl_to_rgb( in vec3 aHSL );

//vec3 srgb_to_linear( in vec3 aSRGB );
//vec3 linear_to_srgb( in vec3 aLinearRGB );

vec3 rgb_to_hsl( in vec3 aRGB )
{
	vec3 hsl = vec3(0.0);

	float channelMin = min( aRGB.r, min( aRGB.g, aRGB.b ) );
	float channelMax = max( aRGB.r, max( aRGB.g, aRGB.b ) );

	float delta = channelMax - channelMin;
	float sum = channelMax + channelMin;

	hsl.b = 0.5*sum;

	if( delta == 0.0 ) 
		return hsl;

	hsl.g = delta / ((hsl.b < 0.5) ? sum : 2.0 - sum);

	vec3 deltaRGB = ((vec3(channelMax)-aRGB) / 6.0 + 0.5 * vec3(channelMax)) / channelMax;

	if( aRGB.r == channelMax ) hsl.r = deltaRGB.b - deltaRGB.g;
	else if( aRGB.g == channelMax ) hsl.r = (1.0/3.0) + deltaRGB.r - deltaRGB.b;
	else if( aRGB.b == channelMax ) hsl.r = (2.0/3.0) + deltaRGB.g - deltaRGB.r;

	if( hsl.r < 0.0 ) hsl.r += 1.0;
	else if( hsl.r > 1.0 ) hsl.r -= 1.0;

	return hsl;
}

float hsl_hue_to_color_( in float aT, in float aU, in float aH )
{
	float h = aH;

	if( h < 0.0 ) h += 1.0;
	else if( h > 1.0 ) h -= 1.0;

	if( 6.0*h < 1.0 ) return aT + (aU-aT) * (6.0*h);
	if( 2.0*h < 1.0 ) return aU;
	if( 3.0*h < 2.0 ) return aT + (aU-aT) * ((2.0/3.0) - h) * 6.0;
	return aT;
}
vec3 hsl_to_rgb( in vec3 aHSL )
{
	if( aHSL.g == 0.0 ) 
		return vec3(aHSL.b);
	
	float t, u;
	if( aHSL.b < 0.5 )
		u = aHSL.b * (1.0+aHSL.g);
	else
		u = aHSL.g + aHSL.b - aHSL.g*aHSL.b;

	t = 2.0 * aHSL.b - u;

	return vec3(
		hsl_hue_to_color_( t, u, aHSL.r + (1.0/3.0) ),
		hsl_hue_to_color_( t, u, aHSL.r ),
		hsl_hue_to_color_( t, u, aHSL.r - (1.0/3.0) )
	);
}

#endif // COLORSPACES_GLSL_34800F2E_2C0A_4035_B227_98599373E023
