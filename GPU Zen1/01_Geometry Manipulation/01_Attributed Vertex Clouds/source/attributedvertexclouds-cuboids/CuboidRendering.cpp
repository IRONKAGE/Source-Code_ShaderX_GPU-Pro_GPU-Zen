
#include "CuboidRendering.h"

#include <iostream>
#include <chrono>
#include <algorithm>

#include <glm/gtc/random.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <glbinding/gl/gl.h>

#include "common.h"

#include "CuboidVertexCloud.h"
#include "CuboidTriangles.h"
#include "CuboidTriangleStrip.h"
#include "CuboidInstancing.h"


using namespace gl;


namespace
{


static const auto gridOffset = 0.2f;

static const auto lightGray = glm::vec3(200) / 255.0f;
static const auto red = glm::vec3(196, 30, 20) / 255.0f;
static const auto orange = glm::vec3(255, 114, 70) / 255.0f;
static const auto yellow = glm::vec3(255, 200, 107) / 255.0f;


} // namespace


CuboidRendering::CuboidRendering()
: Rendering("Cuboids")
, m_gradientTexture(0)
{
}

CuboidRendering::~CuboidRendering()
{
}

void CuboidRendering::onInitialize()
{
    addImplementation(new CuboidTriangles);
    addImplementation(new CuboidTriangleStrip);
    addImplementation(new CuboidInstancing);
    addImplementation(new CuboidVertexCloud);

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

void CuboidRendering::onDeinitialize()
{
    glDeleteTextures(1, &m_gradientTexture);
}

void CuboidRendering::onCreateGeometry()
{
    const auto cuboidGridSize = static_cast<std::size_t>(m_gridSize);
    const auto cuboidCount = static_cast<std::size_t>(cuboidGridSize * cuboidGridSize * cuboidGridSize);
    const auto worldScale = glm::vec3(1.0f) / glm::vec3(cuboidGridSize, cuboidGridSize, cuboidGridSize);

    for (auto implementation : m_implementations)
    {
        implementation->resize(cuboidCount);
    }

    std::array<std::vector<float>, 4> noise;
    for (auto i = size_t(0); i < noise.size(); ++i)
    {
        noise[i] = loadNoise("/noise-"+std::to_string(cuboidGridSize)+"-"+std::to_string(i)+".raw");
    }

#pragma omp parallel for
    for (size_t i = 0; i < cuboidCount; ++i)
    {
        const auto position = glm::ivec3(i % cuboidGridSize, (i / cuboidGridSize) % cuboidGridSize, i / cuboidGridSize / cuboidGridSize);
        const auto offset = glm::vec3(
            (position.y + position.z) % 2 ? gridOffset : 0.0f,
            (position.x + position.z) % 2 ? gridOffset : 0.0f,
            (position.x + position.y) % 2 ? gridOffset : 0.0f
        );

        Cuboid c;
        c.center = glm::vec3(-0.5f, -0.5f, -0.5f) + (glm::vec3(position) + offset) * worldScale;
        c.extent = glm::mix(glm::vec3(0.2f, 0.2f, 0.2f), glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(noise[0][i], noise[1][i], noise[2][i])) * worldScale;
        c.colorValue = glm::mix(0.0f, 1.0f, noise[3][i]);

        for (auto implementation : m_implementations)
        {
            static_cast<CuboidImplementation*>(implementation)->setCube(i, c);
        }
    }
}

void CuboidRendering::onPrepareRendering()
{
    GLuint program = m_current->program();
    const auto gradientSamplerLocation = glGetUniformLocation(program, "gradient");
    glUseProgram(program);
    glUniform1i(gradientSamplerLocation, 0);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_1D, m_gradientTexture);
}

void CuboidRendering::onFinalizeRendering()
{
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_1D, 0);
}
