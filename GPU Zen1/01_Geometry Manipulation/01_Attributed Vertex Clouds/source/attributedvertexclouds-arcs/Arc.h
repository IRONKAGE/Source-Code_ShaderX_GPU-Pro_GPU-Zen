
#pragma once

#include <glm/vec2.hpp>


class Arc
{
public:
    Arc();

    glm::vec2 center;
    glm::vec2 heightRange;
    glm::vec2 angleRange;
    glm::vec2 radiusRange;
    float colorValue;
    int tessellationCount;
};
