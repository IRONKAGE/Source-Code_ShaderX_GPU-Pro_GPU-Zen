#version 330

uniform mat4 viewProjection;

in vec3  in_vertex;
in vec3  in_normal;
in ivec4 in_positionAndType;

uniform float blockSize;

flat out vec3 g_normal;
flat out int  g_type;
     out vec3 g_localCoord;

void main()
{
    gl_Position = viewProjection * vec4((in_vertex + vec3(in_positionAndType.xyz)) * blockSize, 1.0);
    
    g_normal = in_normal;
    g_type = in_positionAndType.w;
    g_localCoord = in_vertex * 2.0;
}
