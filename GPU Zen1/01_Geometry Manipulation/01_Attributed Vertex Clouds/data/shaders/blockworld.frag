#version 330

flat in vec3 g_normal;
flat in int g_type;
in vec3 g_localCoord;

uniform sampler2DArray terrain;
uniform int blockThreshold;

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

out vec3 out_color;

vec2 extract(in vec3 coords, in vec3 mask)
{
    return mix(mix(
            coords.xy,
            coords.xz,
            float(abs(mask.y) > 0.5)
        ),
        coords.yz,
        float(abs(mask.x) > 0.5)
    );
}

void main()
{
    if (g_type <= blockThreshold)
    {
        discard;
        return;
    }
    
    vec3 col = vec3(0.0);
    vec3 N = normalize(g_normal);
    vec2 texCoord = extract(g_localCoord, N) * 0.5 + 0.5;
    vec3 terrainColor = texture(terrain, vec3(texCoord, (g_type-1)/4)).rgb;

    for (int i = 0; i < 6; ++i)
    {
        vec3 L = lightDirs[i];
        float lambertTerm = dot(N,L);

        col += max(lambertTerm, 0.0) * terrainColor * lightStrengths[i];
    }
    
    out_color = col;
    //out_color = N * 0.5 + 0.5;
}
