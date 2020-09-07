
#include "PolygonRendering.h"

#include <iostream>
#include <chrono>
#include <algorithm>

#include <glm/gtc/random.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <glbinding/gl/gl.h>

#include "common.h"

#include "PolygonVertexCloud.h"
#include "PolygonTriangles.h"
#include "PolygonTriangleStrip.h"
//#include "PolygonInstancing.h"


using namespace gl;


namespace
{


static const auto gridOffset = 0.2f;

static const auto lightGray = glm::vec3(200) / 255.0f;
static const auto red = glm::vec3(196, 30, 20) / 255.0f;
static const auto orange = glm::vec3(255, 114, 70) / 255.0f;
static const auto yellow = glm::vec3(255, 200, 107) / 255.0f;


} // namespace


PolygonRendering::PolygonRendering()
: Rendering("Polygons")
, m_gradientTexture(0)
{
}

PolygonRendering::~PolygonRendering()
{
}

void PolygonRendering::onInitialize()
{
    addImplementation(new PolygonTriangles);
    addImplementation(new PolygonTriangleStrip);
    addImplementation(new PolygonVertexCloud);

    glGenTextures(1, &m_gradientTexture);

    std::array<glm::vec3, 4> gradient = {{
        red,
        orange,
        yellow,
        lightGray
    }};

    glBindTexture(GL_TEXTURE_1D, m_gradientTexture);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGB32F, gradient.size(), 0, GL_RGB, GL_FLOAT, gradient.data());
    glBindTexture(GL_TEXTURE_1D, 0);
}

void PolygonRendering::onDeinitialize()
{
    glDeleteTextures(1, &m_gradientTexture);
}

void PolygonRendering::onCreateGeometry()
{
    const auto polygonGridSize = static_cast<std::size_t>(m_gridSize);
    const auto polygonCount = static_cast<std::size_t>(polygonGridSize * polygonGridSize * polygonGridSize);
    const auto worldScale = glm::vec3(1.0f) / glm::vec3(polygonGridSize, polygonGridSize, polygonGridSize);

    for (auto implementation : m_implementations)
    {
        implementation->resize(polygonCount);
    }

    std::array<std::vector<float>, 4> noise;
    for (auto i = size_t(0); i < noise.size(); ++i)
    {
        noise[i] = loadNoise("/noise-"+std::to_string(polygonGridSize)+"-"+std::to_string(i)+".raw");
    }

//#pragma omp parallel for
    for (size_t i = 0; i < polygonCount; ++i)
    {
        const auto position = glm::ivec3(i % polygonGridSize, (i / polygonGridSize) % polygonGridSize, i / polygonGridSize / polygonGridSize);
        const auto offset = glm::vec3(
            (position.y + position.z) % 2 ? gridOffset : 0.0f,
            (position.x + position.z) % 2 ? gridOffset : 0.0f,
            (position.x + position.y) % 2 ? gridOffset : 0.0f
        );

        Polygon p;

        p.heightRange.x = -0.5f + (position.y + offset.y) * worldScale.y - 0.5f * noise[0][i] * worldScale.y;
        p.heightRange.y = -0.5f + (position.y + offset.y) * worldScale.y + 0.5f * noise[0][i] * worldScale.y;

        const auto vertexCount = size_t(3) + size_t(glm::ceil(12.0f * noise[1][i]));
        const auto center = glm::vec2(-0.5f, -0.5f) + (glm::vec2(position.x, position.z) + glm::vec2(offset.x, offset.z)) * glm::vec2(worldScale.x, worldScale.z);
        const auto radius = 0.5f * 0.5f * (noise[2][i] + 1.0f);

        p.points.resize(vertexCount);

        for (auto j = size_t(0); j < vertexCount; ++j)
        {
            const auto angle = glm::pi<float>() * 2.0f * float(j) / float(vertexCount);
            const auto normalizedPosition = glm::vec2(
                glm::cos(angle),
                glm::sin(angle)
            );

            p.points[j] = center + glm::vec2(radius, radius) * normalizedPosition * glm::vec2(worldScale.x, worldScale.z);
        }

        p.colorValue = noise[3][i];

        for (auto implementation : m_implementations)
        {
            static_cast<PolygonImplementation*>(implementation)->setPolygon(i, p);
        }
    }
}

void PolygonRendering::onPrepareRendering()
{
    GLuint program = m_current->program();
    const auto gradientSamplerLocation = glGetUniformLocation(program, "gradient");
    glUseProgram(program);
    glUniform1i(gradientSamplerLocation, 0);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_1D, m_gradientTexture);
}

void PolygonRendering::onFinalizeRendering()
{
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_1D, 0);
}
