#version 400

layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

uniform mat4 viewProjection;

in Vertex
{
    vec3 color;
} vertex[];

flat out vec3 g_color;
flat out vec3 g_normal;

void main()
{
    vec3 normal = cross(gl_in[2].gl_Position.xyz - gl_in[0].gl_Position.xyz, gl_in[1].gl_Position.xyz - gl_in[0].gl_Position.xyz);
    
    g_color = vertex[0].color;
    g_normal = normal;
    gl_Position = viewProjection * gl_in[1].gl_Position;
    EmitVertex();
    
    g_color = vertex[0].color;
    g_normal = normal;
    gl_Position = viewProjection * gl_in[0].gl_Position;
    EmitVertex();
    
    g_color = vertex[0].color;
    g_normal = normal;
    gl_Position = viewProjection * gl_in[2].gl_Position;
    EmitVertex();
    
    EndPrimitive();
}
