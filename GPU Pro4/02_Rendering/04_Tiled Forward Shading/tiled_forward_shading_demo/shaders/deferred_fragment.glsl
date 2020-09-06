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
#include "ObjModel.glsl"

#define SHADING_MODEL_CONSTANTS_ONLY
#include "ShadingModel.glsl"

in vec3 v2f_normal;
in vec3 v2f_tangent;
in vec3 v2f_bitangent;
in vec2 v2f_texCoord;

out vec4 outDiffuse; // WARNING: outDiffuse.w = shading model ID
out vec4 outSpecularShininess;
out vec4 outNormal;
out vec4 outAmbient;

void main()
{
#if ENABLE_ALPHA_TEST
  // Manual alpha test (note: alpha test is no longer part of Opengl 3.3).
  if (texture2D(opacity_texture, v2f_texCoord).r < 0.5)
  {
    discard;
  }
#endif // ENABLE_ALPHA_TEST

  vec3 normal = normalize(v2f_normal);
  vec3 tangent = normalize(v2f_tangent);
  vec3 bitangent = normalize(v2f_bitangent);
  vec3 normalMapSample = texture2D(normal_texture, v2f_texCoord).xyz * vec3(2.0) - vec3(1.0);

  vec3 diffuse = texture2D(diffuse_texture, v2f_texCoord).rgb * material_diffuse_color;
  vec3 specular = texture2D(specular_texture, v2f_texCoord).rgb * material_specular_color;

  outDiffuse = vec4( diffuse, float(SHADING_MODEL) );
  outSpecularShininess = vec4(specular, material_specular_exponent);

  outNormal = vec4(normalize(normalMapSample.x * tangent + normalMapSample.y * bitangent + normalMapSample.z * normal), 1.0);
  // Note: emissive could be included here.
  outAmbient = vec4(diffuse * ambientGlobal, material_alpha);
}
