#version 330

layout (points) in;
layout (triangle_strip, max_vertices = 14) out;

uniform mat4 viewProjection;

in vec2 v_extent[];
in vec3 v_color[];
in float v_height[];

flat out vec3 g_color;
flat out vec3 g_normal;

const vec3 NEGATIVE_X = vec3(-1.0, 0.0, 0.0);
const vec3 NEGATIVE_Y = vec3(0.0, -1.0, 0.0);
const vec3 NEGATIVE_Z = vec3(0.0, 0.0, -1.0);
const vec3 POSITIVE_X = vec3(1.0, 0.0, 0.0);
const vec3 POSITIVE_Y = vec3(0.0, 1.0, 0.0);
const vec3 POSITIVE_Z = vec3(0.0, 0.0, 1.0);

// is called up to 12 times,
// each one with the world position of the current vertex and it's normal (regarding the provoking vertex)
void emit(in vec4 position, in vec3 normal)
{
    gl_Position = viewProjection * position;
    g_normal = normal;
    g_color = v_color[0];
    
    EmitVertex();
}

void generateCuboid(in vec3 center, in vec3 scale)
{
    if (scale.x <= 0.0 || scale.z <= 0.0)
    {
        return;
    }
    
    vec3 llf = center - (vec3(scale.x, scale.y, scale.z) / vec3(2.0));
    vec3 urb = center + (vec3(scale.x, scale.y, scale.z) / vec3(2.0));

    vec4 vertices[8];
    vertices[0] = vec4(llf.x, urb.y, llf.z, 1.0); // A = H
    vertices[1] = vec4(llf.x, urb.y, urb.z, 1.0); // B = F
    vertices[2] = vec4(urb.x, urb.y, llf.z, 1.0); // C = J
    vertices[3] = vec4(urb.x, urb.y, urb.z, 1.0); // D
    vertices[4] = vec4(urb.x, llf.y, urb.z, 1.0); // E = L
    vertices[5] = vec4(llf.x, llf.y, urb.z, 1.0); // G
    vertices[6] = vec4(llf.x, llf.y, llf.z, 1.0); // I
    vertices[7] = vec4(urb.x, llf.y, llf.z, 1.0); // K
    
    emit(vertices[0], POSITIVE_Y); // A
    emit(vertices[1], POSITIVE_Y); // B
    emit(vertices[2], POSITIVE_Y); // C
    emit(vertices[3], POSITIVE_Y); // D
    
    if (scale.y > 0.0)
    {
        emit(vertices[4], POSITIVE_X); // E

        emit(vertices[1], POSITIVE_Z); // F
        emit(vertices[5], POSITIVE_Z); // G

        emit(vertices[0], NEGATIVE_X); // H
        emit(vertices[6], NEGATIVE_X); // I

        emit(vertices[2], NEGATIVE_Z); // J
        emit(vertices[7], NEGATIVE_Z); // K

        emit(vertices[4], POSITIVE_X); // L
    }
    
    emit(vertices[6], NEGATIVE_Y); // I
    emit(vertices[5], NEGATIVE_Y); // G
    
    EndPrimitive();
}

void main()
{
    vec3 center = gl_in[0].gl_Position.xyz;
    center.y += v_height[0] / 2.0;
    vec3 scale = vec3(v_extent[0].x, v_height[0], v_extent[0].y);
    
    generateCuboid(center, scale);
}
