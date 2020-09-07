#version 430

layout (points) in;
layout (triangle_strip, max_vertices = 6) out;

const vec3 UP = vec3(0.0, 1.0, 0.0);
const vec3 DOWN = vec3(0.0, -1.0, 0.0);

uniform mat4 viewProjection;

uniform samplerBuffer centerAndHeights;
uniform samplerBuffer colorValues;

uniform sampler1D gradient;

in vec2 v_start[];
in int v_startIndex[];
in vec2 v_end[];
in int v_endIndex[];

flat out vec3 g_color;
flat out vec3 g_normal;

void emit(in vec3 pos, in vec3 n, in vec3 color)
{
    gl_Position = viewProjection * vec4(pos, 1.0);

    g_color = color;
    g_normal = n;
    
    EmitVertex();
}

void main()
{
    // Handle attributes for mixed prismas due to overlapping VAO configuration
    if (v_startIndex[0] != v_endIndex[0])
    {
        return;
    }
    
    vec4 centerAndHeight = texelFetch(centerAndHeights, v_startIndex[0]).rgba;
    float colorValue = texelFetch(colorValues, v_startIndex[0]).r;
    
    vec3 color = texture(gradient, colorValue).rgb;
    
    vec3 cBottom = vec3(centerAndHeight.r, centerAndHeight.b, centerAndHeight.g);
    vec3 sBottom = vec3(v_start[0].x, centerAndHeight.b, v_start[0].y);
    vec3 eBottom = vec3(v_end[0].x, centerAndHeight.b, v_end[0].y);
    vec3 cTop = vec3(centerAndHeight.r, centerAndHeight.a, centerAndHeight.g);
    vec3 sTop = vec3(v_start[0].x, centerAndHeight.a, v_start[0].y);
    vec3 eTop = vec3(v_end[0].x, centerAndHeight.a, v_end[0].y);
        
    vec3 normal = cross(eBottom - sBottom, UP);
    
    emit(cBottom, DOWN, color);
    emit(sBottom, DOWN, color);
    emit(eBottom, DOWN, color);
    emit(sTop, normal, color);
    emit(eTop, normal, color);
    emit(cTop, UP, color);
    
    EndPrimitive();
}
