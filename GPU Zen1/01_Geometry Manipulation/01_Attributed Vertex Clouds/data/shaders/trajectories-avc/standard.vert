#version 330

layout (location =  0) in vec3  in_position;
layout (location =  1) in int   in_trajectoryID;
layout (location =  2) in int   in_type;
layout (location =  3) in float in_colorValue;
layout (location =  4) in float in_sizeValue;

layout (location =  5) in vec3  prev_position;
layout (location =  6) in int   prev_trajectoryID;
layout (location =  7) in int   prev_type;
layout (location =  8) in float prev_colorValue;
layout (location =  9) in float prev_sizeValue;

layout (location = 10) in vec3  next_position;
layout (location = 11) in int   next_trajectoryID;
layout (location = 12) in int   next_type;
layout (location = 13) in float next_colorValue;
layout (location = 14) in float next_sizeValue;

uniform sampler1D gradient;

out CurrentSegment
{
    vec3  position;
    int   trajectoryID;
    int   type;
    vec3  color;
    float sizeValue;
} current;

out PreviousSegment
{
    vec3  position;
    int   trajectoryID;
    int   type;
    vec3  color;
    float sizeValue;
} previous;

out NextSegment
{
    vec3  position;
    int   trajectoryID;
    int   type;
    vec3  color;
    float sizeValue;
} next;

void main()
{
    current.position = in_position;
    current.trajectoryID = in_trajectoryID;
    current.type = in_type;
    current.color = texture(gradient, in_colorValue).rgb;
    current.sizeValue = in_sizeValue;
    
    previous.position = prev_position;
    previous.trajectoryID = prev_trajectoryID;
    previous.type = prev_type;
    previous.color = texture(gradient, prev_colorValue).rgb;
    previous.sizeValue = prev_sizeValue;
    
    next.position = next_position;
    next.trajectoryID = next_trajectoryID;
    next.type = next_type;
    next.color = texture(gradient, next_colorValue).rgb;
    next.sizeValue = next_sizeValue;
}
