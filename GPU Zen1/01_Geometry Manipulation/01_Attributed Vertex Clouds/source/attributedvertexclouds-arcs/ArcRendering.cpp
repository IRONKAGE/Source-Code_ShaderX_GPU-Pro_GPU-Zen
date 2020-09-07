
#include "ArcRendering.h"

#include <iostream>
#include <chrono>
#include <algorithm>

#include <glm/gtc/random.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <glbinding/gl/gl.h>

#include "common.h"

#include "ArcVertexCloud.h"
//#include "ArcTriangles.h"
//#include "ArcTriangleStrip.h"
//#include "ArcInstancing.h"


using namespace gl;


namespace
{


static const auto arcTessellationCount = size_t(128);
static const auto gridOffset = 0.2f;

static const auto lightGray = glm::vec3(200) / 255.0f;
static const auto red = glm::vec3(196, 30, 20) / 255.0f;
static const auto orange = glm::vec3(255, 114, 70) / 255.0f;
static const auto yellow = glm::vec3(255, 200, 107) / 255.0f;


} // namespace


ArcRendering::ArcRendering()
: Rendering("Arcs")
, m_gradientTexture(0)
{
}

ArcRendering::~ArcRendering()
{
}

void ArcRendering::onInitialize()
{
    addImplementation(new ArcVertexCloud(false));
    addImplementation(new ArcVertexCloud(true));

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

void ArcRendering::onDeinitialize()
{
    glDeleteTextures(1, &m_gradientTexture);
}

void ArcRendering::onCreateGeometry()
{
    const auto arcGridSize = static_cast<std::size_t>(m_gridSize);
    const auto arcCount = arcGridSize * arcGridSize * arcGridSize;
    const auto worldScale = glm::vec3(1.0f) / glm::vec3(arcGridSize, arcGridSize, arcGridSize);


    for (auto implementation : m_implementations)
    {
        implementation->resize(arcCount);
    }

    std::array<std::vector<float>, 7> noise;
    for (auto i = size_t(0); i < noise.size(); ++i)
    {
        noise[i] = loadNoise("/noise-"+std::to_string(arcGridSize)+"-"+std::to_string(i)+".raw");
    }

//#pragma omp parallel for
    for (size_t i = 0; i < arcCount; ++i)
    {
        const auto position = glm::ivec3(i % arcGridSize, (i / arcGridSize) % arcGridSize, i / arcGridSize / arcGridSize);
        const auto offset = glm::vec3(
            (position.y + position.z) % 2 ? gridOffset : 0.0f,
            (position.x + position.z) % 2 ? gridOffset : 0.0f,
            (position.x + position.y) % 2 ? gridOffset : 0.0f
        );

        Arc a;
        a.center = glm::vec2(-0.5f, -0.5f) + (glm::vec2(position.x, position.z) + glm::vec2(offset.x, offset.z)) * glm::vec2(worldScale.x, worldScale.z);

        a.heightRange.x = -0.5f + (position.y + offset.y - 0.5f * noise[0][i]) * worldScale.y;
        a.heightRange.y = -0.5f + (position.y + offset.y + 0.5f * noise[0][i]) * worldScale.y;

        a.angleRange.x = -0.5f * glm::pi<float>() + 0.75f * glm::pi<float>() * noise[1][i];
        a.angleRange.y = 0.25f * glm::pi<float>() + 0.5f * glm::pi<float>() * noise[2][i];

        a.radiusRange.x = 0.3f * noise[3][i] * worldScale.x;
        a.radiusRange.y = a.radiusRange.x + 0.5f * noise[4][i] * worldScale.x;

        a.colorValue = noise[5][i];

        a.tessellationCount = glm::round(1.0f / worldScale.x * (a.angleRange.y - a.angleRange.x) * a.radiusRange.y * glm::mix(4.0f, 64.0f, noise[6][i]) / (2.0f * glm::pi<float>()));

        for (auto implementation : m_implementations)
        {
            static_cast<ArcImplementation*>(implementation)->setArc(i, a);
        }
    }
}

void ArcRendering::onPrepareRendering()
{
    GLuint program = m_current->program();
    const auto gradientSamplerLocation = glGetUniformLocation(program, "gradient");
    glUseProgram(program);
    glUniform1i(gradientSamplerLocation, 0);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_1D, m_gradientTexture);
}

void ArcRendering::onFinalizeRendering()
{
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_1D, 0);
}
