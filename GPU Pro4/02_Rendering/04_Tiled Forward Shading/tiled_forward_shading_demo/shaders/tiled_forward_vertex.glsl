#version 330 compatibility

#include "globals.glsl"

in vec3 position;
in vec3 normalIn;
in vec2 texCoordIn;
in vec3 tangentIn;
in vec3 bitangentIn;

out	vec3 v2f_normal;
out	vec3 v2f_tangent;
out	vec3 v2f_bitangent;
out	vec3 v2f_position;
out	vec2 v2f_texCoord;

void main()
{	
  v2f_normal = normalize((normalMatrix * vec4(normalIn, 0.0)).xyz);
  v2f_tangent = normalize((normalMatrix * vec4(tangentIn, 0.0)).xyz);
  v2f_bitangent = normalize((normalMatrix * vec4(bitangentIn, 0.0)).xyz);
  v2f_texCoord = texCoordIn;
  v2f_position = (viewMatrix * vec4(position, 1.0)).xyz;
  gl_Position = viewProjectionMatrix * vec4(position, 1.0);
} 
