#version 150

in vec2 v_texCoord;

uniform sampler2D colorBuffer;
uniform sampler2D depthBuffer;

out vec3 out_color;

float depthAt(
    const in vec2 coord,
    const in sampler2D depthTexture)
{
    return texture(depthTexture, coord).x;
}

void calculateAO(
    const in vec2 tc,
    const in float threshold,
    const in float aoCap,
    const in float aoMultiplier,
    const in float depth,
    const in sampler2D depthTexture,
    inout float ao,
    inout int pixelsCaculated)
{
    float d = depthAt(tc, depthTexture);

    if (abs(d-depth) < threshold)
    {
        // map to AO amount
        ao += min(aoCap, max(0.0, depth - d) * aoMultiplier);
        // propagate to sum
        ++pixelsCaculated;
    }
}

const vec2 poisson[16] = vec2[16](
    vec2(+0.007937789, +0.73124397),
    vec2(-0.10177308,  -0.6509396),
    vec2(-0.9906806,   -0.63400936),
    vec2(+0.96427417,  -0.25506377),
    vec2(+0.7163085,   +0.22836149),
    vec2(-0.65210974,  +0.37117887),
    vec2(-0.12714535,  +0.112056136),
    vec2(+0.48898065,  -0.66669613),
    vec2(-0.9744036,   +0.9155904),
    vec2(+0.9274436,   -0.9896486),
    vec2(+0.9782181,   +0.90990245),
    vec2(-0.5583586,   -0.3614012),
    vec2(-0.5021933,   -0.9712455),
    vec2(+0.3091557,   -0.17652994),
    vec2(+0.4665941,   +0.96454906),
    vec2(-0.461774,    +0.9360856));

float unsharpmaskingValue(
    const in vec2 texCoord,
    const in sampler2D depthTexture)
{
    float ao = 0.0;
    vec2 t = vec2(2.0) / vec2(textureSize(depthTexture, 0));
    float threshold = 1.5;

    int pixelsCaculated = 0;
    float aoCap = 0.50;
    float aoMultiplier= 5000.0;
    float depth = depthAt(texCoord, depthTexture);

    int iterations = 6;
    int kernelSize = 16;

    if (depth != 1.0) {
        for (int i = 0; i < iterations; i++) {

            // Iterate over kernel
            for (int j = 0; j < kernelSize; j++) {
                calculateAO(
                    texCoord + poisson[j] * t,
                    threshold,
                    aoCap,
                    aoMultiplier,
                    depth,
                    depthTexture,
                    ao,
                    pixelsCaculated);
            }

            t *= 2.0;
            aoMultiplier *= 1.5;
        }

        ao /= float(pixelsCaculated);
    }

    return 1.15 - ao;
}

void main()
{
    vec3 color = texture(colorBuffer, v_texCoord).rgb;
    
    color *= unsharpmaskingValue(v_texCoord, depthBuffer);
    
    out_color = color;
}
