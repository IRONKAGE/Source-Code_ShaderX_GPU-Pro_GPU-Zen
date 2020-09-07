#version 150

in vec2 in_start;
in int in_startIndex;
in vec2 in_end;
in int in_endIndex;

out vec2 v_start;
out int v_startIndex;
out vec2 v_end;
out int v_endIndex;

void main()
{
    v_start = in_start;
    v_startIndex = in_startIndex;
    v_end = in_end;
    v_endIndex = in_endIndex;
    
    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
}
