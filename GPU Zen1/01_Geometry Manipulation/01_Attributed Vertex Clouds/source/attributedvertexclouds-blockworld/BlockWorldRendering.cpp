
#include "BlockWorldRendering.h"

#include <iostream>
#include <chrono>
#include <algorithm>

#include <glm/gtc/random.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <glbinding/gl/gl.h>

#include "common.h"

#include "BlockWorldVertexCloud.h"
#include "BlockWorldTriangles.h"
#include "BlockWorldTriangleStrip.h"
#include "BlockWorldInstancing.h"


using namespace gl;


BlockWorldRendering::BlockWorldRendering()
: Rendering("BlockWorld")
, m_terrainTexture(0)
, m_blockThreshold(7)
{
}

BlockWorldRendering::~BlockWorldRendering()
{
}

void BlockWorldRendering::onInitialize()
{
    addImplementation(new BlockWorldTriangles);
    addImplementation(new BlockWorldTriangleStrip);
    addImplementation(new BlockWorldInstancing);
    addImplementation(new BlockWorldVertexCloud);

    glGenTextures(1, &m_terrainTexture);

    glBindTexture(GL_TEXTURE_2D_ARRAY, m_terrainTexture);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, static_cast<GLint>(GL_LINEAR_MIPMAP_LINEAR));
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, static_cast<GLint>(GL_LINEAR));
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, static_cast<GLint>(GL_MIRRORED_REPEAT));
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, static_cast<GLint>(GL_MIRRORED_REPEAT));

    auto terrainData = rawFromFile("data/textures/terrain.512.2048.rgba.ub.raw");
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, static_cast<GLint>(GL_RGBA8), 512, 512, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, terrainData.data());

    glGenerateMipmap(GL_TEXTURE_2D_ARRAY);

    glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
}

void BlockWorldRendering::onDeinitialize()
{
    glDeleteTextures(1, &m_terrainTexture);
}

void BlockWorldRendering::onCreateGeometry()
{
    const auto blockGridSize = m_gridSize;
    const auto blockCount = static_cast<std::size_t>(blockGridSize * blockGridSize * blockGridSize);
    const auto worldScale = glm::vec3(1.0f) / glm::vec3(blockGridSize, blockGridSize, blockGridSize);

    for (auto implementation : m_implementations)
    {
        implementation->resize(blockCount);
        static_cast<BlockWorldImplementation*>(implementation)->setBlockSize(worldScale.x);
    }

    std::array<std::vector<float>, 1> noise;
    for (auto i = size_t(0); i < noise.size(); ++i)
    {
        noise[i] = loadNoise("/noise-"+std::to_string(blockGridSize)+"-"+std::to_string(i)+".raw");
    }

#pragma omp parallel for
    for (size_t i = 0; i < blockCount; ++i)
    {
        const auto position = glm::ivec3(i % blockGridSize, (i / blockGridSize) % blockGridSize, i / blockGridSize / blockGridSize) - glm::ivec3(blockGridSize / 2, blockGridSize / 2, blockGridSize / 2);

        Block b;
        b.position = position;
        b.type = static_cast<int>(glm::round(16.0f * noise[0][i]));

        for (auto implementation : m_implementations)
        {
            static_cast<BlockWorldImplementation*>(implementation)->setBlock(i, b);
        }
    }
}

void BlockWorldRendering::onPrepareRendering()
{
    GLuint program = m_current->program();
    const auto terrainSamplerLocation = glGetUniformLocation(program, "terrain");
    const auto blockThresholdLocation = glGetUniformLocation(program, "blockThreshold");
    glUseProgram(program);
    glUniform1i(terrainSamplerLocation, 0);
    glUniform1i(blockThresholdLocation, m_blockThreshold);

    glUseProgram(0);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D_ARRAY, m_terrainTexture);
}

void BlockWorldRendering::onFinalizeRendering()
{
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
}

void BlockWorldRendering::increaseBlockThreshold()
{
    m_blockThreshold = glm::min(m_blockThreshold+1, 14);
}

void BlockWorldRendering::decreaseBlockThreshold()
{
    m_blockThreshold = glm::max(m_blockThreshold-1, 0);
}
