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

#extension GL_EXT_shader_image_load_store : enable

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
#endif // NUM_MSAA_SAMPLES == 1


layout(size2x32) uniform imageBuffer diffuseImage;
layout(size2x32) uniform imageBuffer specularShininessImage;
layout(size2x32) uniform imageBuffer ambientImage;
layout(size2x32) uniform imageBuffer normalImage;
layout(size1x32) uniform imageBuffer depthImage;


out vec4 resultColor;

/**
 * This shader simply shovels all samples of a set of MSAA G-Buffers into generic buffers, via
 * imageStore. 
 * This is used in the demo to get the G-Buffer data into buffers that are able to be mapped into
 * CUDA. There is, sadly, no way to access MSAA textures in CUDA (as of CUDA 4.2).
 */
void main()
{
	ivec2 fragLoc = ivec2(gl_FragCoord.xy);

	int offset = (fragLoc.y * fbSize.x + fragLoc.x) * NUM_MSAA_SAMPLES;

  for (int sampleIndex = 0; sampleIndex < NUM_MSAA_SAMPLES; ++sampleIndex)
  {
    vec4 diffuse = texelFetch(diffuseTex, fragLoc, sampleIndex);
    vec4 specularShininess = texelFetch(specularShininessTex, fragLoc, sampleIndex); 
    vec4 ambient = texelFetch(ambientTex, fragLoc, sampleIndex); 
		float depth = texelFetch(depthTex, fragLoc, sampleIndex).x;
    vec4 normal = texelFetch(normalTex, fragLoc, sampleIndex); 

		imageStore(diffuseImage, offset + sampleIndex, diffuse);
		imageStore(specularShininessImage, offset + sampleIndex, specularShininess);
		imageStore(ambientImage, offset + sampleIndex, ambient);
		imageStore(depthImage, offset + sampleIndex, vec4(depth));
		imageStore(normalImage, offset + sampleIndex, normal);
  }

	resultColor = vec4(1.0);
}
