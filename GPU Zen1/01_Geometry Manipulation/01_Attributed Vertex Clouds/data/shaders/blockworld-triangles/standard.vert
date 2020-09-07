#version 330

uniform mat4 viewProjection;

in vec3 in_vertex;
in vec3 in_normal;
in vec3 in_localCoord;
in int  in_type;

flat out vec3 g_normal;
flat out int  g_type;
     out vec3 g_localCoord;

void main()
{
    gl_Position = viewProjection * vec4(in_vertex, 1.0);
    
    g_normal = in_normal;
    g_type = in_type;
    g_localCoord = in_localCoord;
}
