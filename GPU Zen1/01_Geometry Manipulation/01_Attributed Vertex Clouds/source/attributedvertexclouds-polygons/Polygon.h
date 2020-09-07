
#pragma once

#include <vector>

#include <glm/vec2.hpp>

class Polygon
{
public:
    Polygon();

    std::vector<glm::vec2> points;
    glm::vec2 heightRange;
    float colorValue;
};
