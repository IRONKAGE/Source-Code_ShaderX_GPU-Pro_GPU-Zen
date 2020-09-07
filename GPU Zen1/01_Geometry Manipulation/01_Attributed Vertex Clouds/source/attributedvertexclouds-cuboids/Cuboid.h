
#pragma once

#include <glm/vec3.hpp>


class Cuboid
{
public:
    Cuboid();

    glm::vec3 center;
    glm::vec3 extent;
    float colorValue;
};
