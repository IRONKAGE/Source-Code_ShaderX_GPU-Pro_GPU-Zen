#version 330

in vec2  in_center;
in vec2  in_extent;
in vec2  in_heightRange;
in float in_colorValue;

uniform sampler1D gradient;

out vec2 v_extent;
out vec3 v_color;
out float v_height;

void main()
{
    gl_Position = vec4(in_center.x, in_heightRange.x, in_center.y, 1.0);
    
    v_extent = in_extent;
    v_color = texture(gradient, in_colorValue).rgb;
    v_height = in_heightRange.y;
}
