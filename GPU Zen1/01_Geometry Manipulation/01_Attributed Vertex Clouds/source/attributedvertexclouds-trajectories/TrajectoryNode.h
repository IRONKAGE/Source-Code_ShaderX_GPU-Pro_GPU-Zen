
#pragma once

#include <glm/vec3.hpp>


class TrajectoryNode
{
public:
    TrajectoryNode();

    int trajectoryID;
    glm::vec3 position;
    int type;
    float colorValue;
    float sizeValue;
};
