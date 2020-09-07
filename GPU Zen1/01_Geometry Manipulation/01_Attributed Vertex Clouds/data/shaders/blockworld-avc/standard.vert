#version 330

in ivec4  in_positionAndType;

out int v_type;

void main()
{
    gl_Position = vec4(in_positionAndType.xyz, 1.0);
    
    v_type = in_positionAndType.w;
}
