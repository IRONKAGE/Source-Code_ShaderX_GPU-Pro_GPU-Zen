
#include "BlockWorldTriangles.h"

#include <algorithm>

#include <glbinding/gl/gl.h>

#include "common.h"

using namespace gl;

BlockWorldTriangles::BlockWorldTriangles()
: BlockWorldImplementation("Triangles")
, m_vertices(0)
, m_vao(0)
, m_vertexShader(0)
, m_fragmentShader(0)
{
}

BlockWorldTriangles::~BlockWorldTriangles()
{
    glDeleteBuffers(1, &m_vertices);
    glDeleteVertexArrays(1, &m_vao);
    glDeleteShader(m_vertexShader);
    glDeleteShader(m_fragmentShader);
    glDeleteProgram(m_program);
}

void BlockWorldTriangles::onInitialize()
{
    glGenBuffers(1, &m_vertices);
    glGenVertexArrays(1, &m_vao);

    initializeVAO();

    m_vertexShader = glCreateShader(GL_VERTEX_SHADER);
    m_fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

    m_program = glCreateProgram();

    glAttachShader(m_program, m_vertexShader);
    glAttachShader(m_program, m_fragmentShader);

    loadShader();
}

void BlockWorldTriangles::initializeVAO()
{
    glBindVertexArray(m_vao);

    glBindBuffer(GL_ARRAY_BUFFER, m_vertices);
    glBufferData(GL_ARRAY_BUFFER, byteSize(), nullptr, GL_STATIC_DRAW);

    glBufferSubData(GL_ARRAY_BUFFER, verticesCount() * sizeof(float) * 0, verticesCount() * sizeof(float) * 3, m_vertex.data());
    glBufferSubData(GL_ARRAY_BUFFER, verticesCount() * sizeof(float) * 3, verticesCount() * sizeof(float) * 3, m_normal.data());
    glBufferSubData(GL_ARRAY_BUFFER, verticesCount() * sizeof(float) * 6, verticesCount() * sizeof(float) * 3, m_localCoords.data());
    glBufferSubData(GL_ARRAY_BUFFER, verticesCount() * sizeof(float) * 9, verticesCount() * sizeof(float) * 1, m_type.data());

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), reinterpret_cast<void*>(verticesCount() * sizeof(float) * 0));
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), reinterpret_cast<void*>(verticesCount() * sizeof(float) * 3));
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), reinterpret_cast<void*>(verticesCount() * sizeof(float) * 6));
    glVertexAttribIPointer(3, 1, GL_INT, sizeof(float), reinterpret_cast<void*>(verticesCount() * sizeof(float) * 9));

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glEnableVertexAttribArray(3);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

bool BlockWorldTriangles::loadShader()
{
    const auto vertexShaderSource = loadShaderSource("/blockworld-triangles/standard.vert");
    const auto vertexShaderSource_ptr = vertexShaderSource.c_str();
    if(vertexShaderSource_ptr)
        glShaderSource(m_vertexShader, 1, &vertexShaderSource_ptr, 0);

    glCompileShader(m_vertexShader);

    bool success = checkForCompilationError(m_vertexShader, "vertex shader");

    const auto fragmentShaderSource = loadShaderSource("/blockworld.frag");
    const auto fragmentShaderSource_ptr = fragmentShaderSource.c_str();
    if(fragmentShaderSource_ptr)
        glShaderSource(m_fragmentShader, 1, &fragmentShaderSource_ptr, 0);

    glCompileShader(m_fragmentShader);

    success &= checkForCompilationError(m_fragmentShader, "fragment shader");


    if (!success)
    {
        return false;
    }

    glLinkProgram(m_program);

    success &= checkForLinkerError(m_program, "program");

    if (!success)
    {
        return false;
    }

    glBindFragDataLocation(m_program, 0, "out_color");

    return true;
}

void BlockWorldTriangles::setBlock(size_t index, const Block & block)
{
    static const glm::vec3 NEGATIVE_X = glm::vec3(-1.0, 0.0, 0.0);
    static const glm::vec3 NEGATIVE_Y = glm::vec3(0.0, -1.0, 0.0);
    static const glm::vec3 NEGATIVE_Z = glm::vec3(0.0, 0.0, -1.0);
    static const glm::vec3 POSITIVE_X = glm::vec3(1.0, 0.0, 0.0);
    static const glm::vec3 POSITIVE_Y = glm::vec3(0.0, 1.0, 0.0);
    static const glm::vec3 POSITIVE_Z = glm::vec3(0.0, 0.0, 1.0);

    static const glm::vec3 vertices[8] = {
        glm::vec3(-0.5f, 0.5f, -0.5f), // A = H
        glm::vec3(-0.5f, 0.5f, 0.5f), // B = F
        glm::vec3(0.5f, 0.5f, -0.5f), // C = J
        glm::vec3(0.5f, 0.5f, 0.5f), // D
        glm::vec3(0.5f, -0.5f, 0.5f), // E = L
        glm::vec3(-0.5f, -0.5f, 0.5f), // G
        glm::vec3(-0.5f, -0.5f, -0.5f), // I
        glm::vec3(0.5f, -0.5f, -0.5f), // K
    };

    m_normal[verticesPerCuboid() * index + 0] = NEGATIVE_X;
    m_normal[verticesPerCuboid() * index + 1] = NEGATIVE_X;
    m_normal[verticesPerCuboid() * index + 2] = NEGATIVE_X;
    m_normal[verticesPerCuboid() * index + 3] = NEGATIVE_X;
    m_normal[verticesPerCuboid() * index + 4] = NEGATIVE_X;
    m_normal[verticesPerCuboid() * index + 5] = NEGATIVE_X;

    m_normal[verticesPerCuboid() * index + 6] = NEGATIVE_Z;
    m_normal[verticesPerCuboid() * index + 7] = NEGATIVE_Z;
    m_normal[verticesPerCuboid() * index + 8] = NEGATIVE_Z;
    m_normal[verticesPerCuboid() * index + 9] = NEGATIVE_Z;
    m_normal[verticesPerCuboid() * index + 10] = NEGATIVE_Z;
    m_normal[verticesPerCuboid() * index + 11] = NEGATIVE_Z;

    m_normal[verticesPerCuboid() * index + 12] = POSITIVE_X;
    m_normal[verticesPerCuboid() * index + 13] = POSITIVE_X;
    m_normal[verticesPerCuboid() * index + 14] = POSITIVE_X;
    m_normal[verticesPerCuboid() * index + 15] = POSITIVE_X;
    m_normal[verticesPerCuboid() * index + 16] = POSITIVE_X;
    m_normal[verticesPerCuboid() * index + 17] = POSITIVE_X;

    m_normal[verticesPerCuboid() * index + 18] = POSITIVE_Z;
    m_normal[verticesPerCuboid() * index + 19] = POSITIVE_Z;
    m_normal[verticesPerCuboid() * index + 20] = POSITIVE_Z;
    m_normal[verticesPerCuboid() * index + 21] = POSITIVE_Z;
    m_normal[verticesPerCuboid() * index + 22] = POSITIVE_Z;
    m_normal[verticesPerCuboid() * index + 23] = POSITIVE_Z;

    m_normal[verticesPerCuboid() * index + 24] = POSITIVE_Y;
    m_normal[verticesPerCuboid() * index + 25] = POSITIVE_Y;
    m_normal[verticesPerCuboid() * index + 26] = POSITIVE_Y;
    m_normal[verticesPerCuboid() * index + 27] = POSITIVE_Y;
    m_normal[verticesPerCuboid() * index + 28] = POSITIVE_Y;
    m_normal[verticesPerCuboid() * index + 29] = POSITIVE_Y;

    m_normal[verticesPerCuboid() * index + 30] = NEGATIVE_Y;
    m_normal[verticesPerCuboid() * index + 31] = NEGATIVE_Y;
    m_normal[verticesPerCuboid() * index + 32] = NEGATIVE_Y;
    m_normal[verticesPerCuboid() * index + 33] = NEGATIVE_Y;
    m_normal[verticesPerCuboid() * index + 34] = NEGATIVE_Y;
    m_normal[verticesPerCuboid() * index + 35] = NEGATIVE_Y;

    m_vertex[verticesPerCuboid() * index + 0] = vertices[1];
    m_vertex[verticesPerCuboid() * index + 1] = vertices[0];
    m_vertex[verticesPerCuboid() * index + 2] = vertices[5];
    m_vertex[verticesPerCuboid() * index + 3] = vertices[5];
    m_vertex[verticesPerCuboid() * index + 4] = vertices[0];
    m_vertex[verticesPerCuboid() * index + 5] = vertices[6];

    m_vertex[verticesPerCuboid() * index + 6] = vertices[6];
    m_vertex[verticesPerCuboid() * index + 7] = vertices[0];
    m_vertex[verticesPerCuboid() * index + 8] = vertices[7];
    m_vertex[verticesPerCuboid() * index + 9] = vertices[7];
    m_vertex[verticesPerCuboid() * index + 10] = vertices[0];
    m_vertex[verticesPerCuboid() * index + 11] = vertices[2];

    m_vertex[verticesPerCuboid() * index + 12] = vertices[2];
    m_vertex[verticesPerCuboid() * index + 13] = vertices[3];
    m_vertex[verticesPerCuboid() * index + 14] = vertices[7];
    m_vertex[verticesPerCuboid() * index + 15] = vertices[7];
    m_vertex[verticesPerCuboid() * index + 16] = vertices[3];
    m_vertex[verticesPerCuboid() * index + 17] = vertices[4];

    m_vertex[verticesPerCuboid() * index + 18] = vertices[4];
    m_vertex[verticesPerCuboid() * index + 19] = vertices[3];
    m_vertex[verticesPerCuboid() * index + 20] = vertices[5];
    m_vertex[verticesPerCuboid() * index + 21] = vertices[5];
    m_vertex[verticesPerCuboid() * index + 22] = vertices[3];
    m_vertex[verticesPerCuboid() * index + 23] = vertices[1];

    m_vertex[verticesPerCuboid() * index + 24] = vertices[1];
    m_vertex[verticesPerCuboid() * index + 25] = vertices[3];
    m_vertex[verticesPerCuboid() * index + 26] = vertices[0];
    m_vertex[verticesPerCuboid() * index + 27] = vertices[0];
    m_vertex[verticesPerCuboid() * index + 28] = vertices[3];
    m_vertex[verticesPerCuboid() * index + 29] = vertices[2];

    m_vertex[verticesPerCuboid() * index + 30] = vertices[4];
    m_vertex[verticesPerCuboid() * index + 31] = vertices[5];
    m_vertex[verticesPerCuboid() * index + 32] = vertices[6];
    m_vertex[verticesPerCuboid() * index + 33] = vertices[6];
    m_vertex[verticesPerCuboid() * index + 34] = vertices[7];
    m_vertex[verticesPerCuboid() * index + 35] = vertices[4];

    for (auto i = 0ull; i < verticesPerCuboid(); ++i)
    {
        m_localCoords[verticesPerCuboid() * index + i] = m_vertex[verticesPerCuboid() * index + i] * 2.0f;
        m_vertex[verticesPerCuboid() * index + i] = (glm::vec3(block.position) + m_vertex[verticesPerCuboid() * index + i]) * m_blockSize;
        m_type[verticesPerCuboid() * index + i] = block.type;
    }
}

size_t BlockWorldTriangles::size() const
{
    return m_vertex.size() / verticesPerCuboid();
}

size_t BlockWorldTriangles::verticesPerCuboid() const
{
    return 36;
}

size_t BlockWorldTriangles::verticesCount() const
{
    return size() * verticesPerCuboid();
}

size_t BlockWorldTriangles::staticByteSize() const
{
    return 0;
}

size_t BlockWorldTriangles::byteSize() const
{
    return verticesCount() * vertexByteSize();
}

size_t BlockWorldTriangles::vertexByteSize() const
{
    return sizeof(float) * componentCount();
}

size_t BlockWorldTriangles::componentCount() const
{
    return 10;
}

void BlockWorldTriangles::resize(size_t count)
{
    m_vertex.resize(count * verticesPerCuboid());
    m_normal.resize(count * verticesPerCuboid());
    m_localCoords.resize(count * verticesPerCuboid());
    m_type.resize(count * verticesPerCuboid());
}

void BlockWorldTriangles::onRender()
{
    glBindVertexArray(m_vao);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    glDisable(GL_CULL_FACE);
    //glCullFace(GL_BACK);

    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDepthMask(GL_TRUE);

    glUseProgram(m_program);
    glDrawArrays(GL_TRIANGLES, 0, verticesCount());

    glDepthMask(GL_TRUE);

    glUseProgram(0);

    glBindVertexArray(0);
}

gl::GLuint BlockWorldTriangles::program() const
{
    return m_program;
}
