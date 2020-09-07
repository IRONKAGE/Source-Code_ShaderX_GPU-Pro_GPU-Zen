
#include "TrajectoryRendering.h"

#include <iostream>
#include <chrono>
#include <algorithm>

#include <glm/gtc/random.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <glbinding/gl/gl.h>

#include "common.h"

#include "TrajectoryVertexCloud.h"


using namespace gl;


namespace
{


static const auto lightGray = glm::vec3(200) / 255.0f;
static const auto red = glm::vec3(196, 30, 20) / 255.0f;
static const auto orange = glm::vec3(255, 114, 70) / 255.0f;
static const auto yellow = glm::vec3(255, 200, 107) / 255.0f;


} // namespace


TrajectoryRendering::TrajectoryRendering()
: Rendering("Trajectories")
, m_gradientTexture(0)
{
}

TrajectoryRendering::~TrajectoryRendering()
{
}

void TrajectoryRendering::onInitialize()
{
    addImplementation(new TrajectoryVertexCloud);

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

void TrajectoryRendering::onDeinitialize()
{
    glDeleteTextures(1, &m_gradientTexture);
}

void TrajectoryRendering::onCreateGeometry()
{
    const auto trajectoryGridSize = static_cast<std::size_t>(m_gridSize);
    const auto trajectoryCount = static_cast<std::size_t>(trajectoryGridSize * trajectoryGridSize);
    const auto worldScale = glm::vec3(1.0f) / glm::vec3(trajectoryGridSize, trajectoryGridSize, trajectoryGridSize);

    for (auto implementation : m_implementations)
    {
        implementation->resize(trajectoryCount);
    }

    std::array<std::vector<float>, 3> noise;
    for (auto i = size_t(0); i < noise.size(); ++i)
    {
        noise[i] = loadNoise("/noise-"+std::to_string(trajectoryGridSize)+"-"+std::to_string(i)+".raw");
    }

#pragma omp parallel for
    for (size_t i = 0; i < trajectoryCount; ++i)
    {
        const auto position = glm::ivec3((i / trajectoryGridSize) % trajectoryGridSize, i % trajectoryGridSize, 1);

        TrajectoryNode t;

        const auto angle = (position.x + position.y * 0.5f) / static_cast<float>(trajectoryGridSize);
        const auto radius = 0.5f + 0.3f * glm::cos(position.x / static_cast<float>(trajectoryGridSize) * 1.0f * glm::pi<float>())
             + 0.1f * glm::sin(position.y / static_cast<float>(trajectoryGridSize) * 1.0f * glm::pi<float>());

        t.position = glm::vec3(0.0f, -0.5f, 0.0f);
        t.position += glm::vec3(
            radius * glm::cos(2.0f * glm::pi<float>() * angle),
            position.y * worldScale.y,
            radius * glm::sin(2.0f * glm::pi<float>() * angle)
        );
        t.trajectoryID = position.x;
        t.type = noise[0][i] > 0.0f ? 2 : 1;
        t.sizeValue = glm::mix(0.3f, 0.9f, noise[1][i]) * worldScale.x;
        t.colorValue = noise[2][i];

        for (auto implementation : m_implementations)
        {
            static_cast<TrajectoryVertexCloud*>(implementation)->setTrajectoryNode(i, t);
        }
    }
}

void TrajectoryRendering::onPrepareRendering()
{
    GLuint program = m_current->program();
    const auto gradientSamplerLocation = glGetUniformLocation(program, "gradient");
    glUseProgram(program);
    glUniform1i(gradientSamplerLocation, 0);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_1D, m_gradientTexture);
}

void TrajectoryRendering::onFinalizeRendering()
{
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_1D, 0);
}
