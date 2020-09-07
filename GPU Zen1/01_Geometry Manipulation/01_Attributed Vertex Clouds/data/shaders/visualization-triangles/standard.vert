#version 330

uniform mat4 viewProjection;

in vec3  in_vertex;
in vec3  in_normal;
in float in_colorValue;

uniform sampler1D gradient;

flat out vec3 g_color;
flat out vec3 g_normal;

void main()
{
    gl_Position = viewProjection * vec4(in_vertex, 1.0);
    
    g_color = texture(gradient, in_colorValue).rgb;
    g_normal = in_normal;
}
