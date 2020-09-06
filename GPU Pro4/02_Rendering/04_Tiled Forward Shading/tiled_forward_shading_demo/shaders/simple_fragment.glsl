#version 330 compatibility
/****************************************************************************/
/* Copyright (c) 2011, Ola Olsson, Ulf Assarsson
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
/****************************************************************************/
#include "globals.glsl"
#include "srgb.glsl"
#include "ObjModel.glsl"

in vec3 v2f_normal;
in vec2 v2f_texCoord;

out vec4 fragmentColor;

void main() 
{
#if ENABLE_ALPHA_TEST
	// Manual alpha test (note: alpha test is no longer part of Opengl 3.3).
	if (texture2D(opacity_texture, v2f_texCoord).r < 0.5)
	{
		discard;
	}
#endif // ENABLE_ALPHA_TEST

	vec3 up = normalize((normalMatrix * vec4(worldUpDirection, 0.0)).xyz);
	vec3 materialDiffuse = texture(diffuse_texture, v2f_texCoord).xyz * material_diffuse_color;
	vec3 color = materialDiffuse * (0.1 + 0.9 * max(0.0, dot(normalize(v2f_normal), up)));
	fragmentColor = vec4(toSrgb(color), material_alpha);
}
