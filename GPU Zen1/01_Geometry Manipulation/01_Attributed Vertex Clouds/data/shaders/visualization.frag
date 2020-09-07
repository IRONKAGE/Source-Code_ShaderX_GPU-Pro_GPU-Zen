#version 150

flat in vec3 g_color;
flat in vec3 g_normal;

out vec3 out_color;

uniform vec3 lightDirs[6] = vec3[](
    vec3( 0,  1,  0),
    vec3( 1,  0,  0),
    vec3(-1,  0,  0),
    vec3( 0,  0,  1),
    vec3( 0,  0, -1),
    vec3( 0, -1,  0)
);

uniform float lightStrengths[6] = float[](
    1.0,
    0.95,
    0.9,
    0.85,
    0.8,
    0.75
);

void main()
{
    vec3 col = vec3(0.0);
    vec3 N = normalize(g_normal);

    for (int i = 0; i < 6; ++i)
    {
        vec3 L = lightDirs[i];
        float lambertTerm = dot(N,L);

        col += max(lambertTerm, 0.0) * g_color * lightStrengths[i];
    }
    
    out_color = col;
}
