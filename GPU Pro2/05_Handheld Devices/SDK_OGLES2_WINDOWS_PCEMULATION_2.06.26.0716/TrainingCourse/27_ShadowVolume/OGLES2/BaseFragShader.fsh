/*
  Simple fragment shader:
  - Single texturing modulated with vertex lighting
*/

uniform sampler2D sTexture;

varying lowp    float LightIntensity;
varying mediump vec2  TexCoord;

void main()
{
    gl_FragColor = vec4(texture2D(sTexture, TexCoord).rgb * LightIntensity, 1.0);
}
