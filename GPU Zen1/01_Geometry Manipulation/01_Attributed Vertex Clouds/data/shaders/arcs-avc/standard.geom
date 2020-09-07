#version 400

layout (lines) in;
layout (triangle_strip, max_vertices = 12) out;
// Assumption: either vertex[0].hasSide or vertex[1].hasSide has side, never both

uniform mat4 viewProjection;

in Vertex
{
    float angle;
    vec2 radiusRange;
    vec2 center;
    vec2 heightRange;
    vec3 color;
    bool hasSide;
} vertex[];

flat out vec3 g_color;
flat out vec3 g_normal;

void emit(in vec3 pos, in vec3 normal)
{
    gl_Position = viewProjection * vec4(pos, 1.0);
    g_color = vertex[0].color;
    g_normal = normal;
    
    EmitVertex();
}

vec3 circlePoint(in float angle, in float radius, in float height)
{
    return vec3(sin(angle), height, cos(angle))
        * vec3(radius, 1.0, radius)
        + vec3(vertex[0].center.x, 0.0, vertex[0].center.y);
}

void main()
{
    vec3 A = circlePoint(vertex[0].angle, vertex[0].radiusRange.x, vertex[0].heightRange.x);
    vec3 B = circlePoint(vertex[1].angle, vertex[0].radiusRange.x, vertex[0].heightRange.x);
    vec3 C = circlePoint(vertex[1].angle, vertex[0].radiusRange.y, vertex[0].heightRange.x);
    vec3 D = circlePoint(vertex[0].angle, vertex[0].radiusRange.y, vertex[0].heightRange.x);
    vec3 E = circlePoint(vertex[0].angle, vertex[0].radiusRange.x, vertex[0].heightRange.y);
    vec3 F = circlePoint(vertex[1].angle, vertex[0].radiusRange.x, vertex[0].heightRange.y);
    vec3 G = circlePoint(vertex[1].angle, vertex[0].radiusRange.y, vertex[0].heightRange.y);
    vec3 H = circlePoint(vertex[0].angle, vertex[0].radiusRange.y, vertex[0].heightRange.y);
    
    vec3 top = vec3(0.0, 1.0, 0.0);
    vec3 bottom = vec3(0.0, -1.0, 0.0);
    bool hasHeight = vertex[0].heightRange.y - vertex[0].heightRange.x > 0.0;

    vec3 left = normalize(cross(E-A, D-A));
    vec3 right = normalize(cross(F-B, C-B));
    vec3 front = normalize(cross(B-A, E-A));
    //vec3 front = normalize(mix(A, B, 0.5) - vec3(vertex[0].center.x, vertex[0].heightRange.x, vertex[0].center.y));
    vec3 back = -front;
    
    if (hasHeight)
    {
        if (vertex[1].hasSide)
        {
            emit(B, right);
            emit(F, right);
            /*emit(C, right);
            emit(G, right);
            EndPrimitive();*/
        }
        
        emit(C, right);
    }
    
    emit(G, right);
    emit(H, back);
    emit(F, top);
    emit(E, top);
    emit(B, front); // B = G if no height
    emit(A, front); // A = H if no height
    
    if (hasHeight)
    {
        emit(C, bottom);
        emit(D, bottom);
        emit(H, back);
        
        if (vertex[0].hasSide)
        {
            /*EndPrimitive();
            emit(D, bottom);
            emit(H, back);*/
            emit(A, left);
            emit(E, left);
        }
    }
    
    EndPrimitive();
}
