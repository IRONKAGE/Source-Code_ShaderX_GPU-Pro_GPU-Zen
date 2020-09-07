
#include "CuboidInstancing.h"

#include <algorithm>

#include <glbinding/gl/gl.h>

#include "common.h"

using namespace gl;

CuboidInstancing::CuboidInstancing()
: CuboidImplementation("Instancing")
, m_vertices(0)
, m_vao(0)
, m_vertexShader(0)
, m_fragmentShader(0)
{
}

CuboidInstancing::~CuboidInstancing()
{
    glDeleteBuffers(1, &m_vertices);
    glDeleteBuffers(1, &m_attributes);
    glDeleteVertexArrays(1, &m_vao);
    glDeleteShader(m_vertexShader);
    glDeleteShader(m_fragmentShader);
    glDeleteProgram(m_program);
}

void CuboidInstancing::onInitialize()
{
    glGenBuffers(1, &m_vertices);
    glGenBuffers(1, &m_attributes);
    glGenVertexArrays(1, &m_vao);

    initializeVAO();

    m_vertexShader = glCreateShader(GL_VERTEX_SHADER);
    m_fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

    m_program = glCreateProgram();

    glAttachShader(m_program, m_vertexShader);
    glAttachShader(m_program, m_fragmentShader);

    loadShader();
}

void CuboidInstancing::initializeVAO()
{
    glBindVertexArray(m_vao);

    glBindBuffer(GL_ARRAY_BUFFER, m_vertices);

    static const auto NEGATIVE_X = glm::vec3(-1.0, 0.0, 0.0);
    static const auto NEGATIVE_Y = glm::vec3(0.0, -1.0, 0.0);
    static const auto NEGATIVE_Z = glm::vec3(0.0, 0.0, -1.0);
    static const auto POSITIVE_X = glm::vec3(1.0, 0.0, 0.0);
    static const auto POSITIVE_Y = glm::vec3(0.0, 1.0, 0.0);
    static const auto POSITIVE_Z = glm::vec3(0.0, 0.0, 1.0);

    const auto vertices = std::array<glm::vec3, 14>{{
        glm::vec3(-1.0f, 1.0f, -1.0f),
        glm::vec3(-1.0f, 1.0f, 1.0f),
        glm::vec3(1.0f, 1.0f, -1.0f),
        glm::vec3(1.0f, 1.0f, 1.0f),
        glm::vec3(1.0f, -1.0f, 1.0f),
        glm::vec3(-1.0f, 1.0f, 1.0f),
        glm::vec3(-1.0f, -1.0f, 1.0f),
        glm::vec3(-1.0f, 1.0f, -1.0f),
        glm::vec3(-1.0f, -1.0f, -1.0f),
        glm::vec3(1.0f, 1.0f, -1.0f),
        glm::vec3(1.0f, -1.0f, -1.0f),
        glm::vec3(1.0f, -1.0f, 1.0f),
        glm::vec3(-1.0f, -1.0f, -1.0f),
        glm::vec3(-1.0f, -1.0f, 1.0f)
    }};

    const auto normals = std::array<glm::vec3, 14>{{
        POSITIVE_Y, POSITIVE_Y, POSITIVE_Y, POSITIVE_Y,
        POSITIVE_X, POSITIVE_Z, POSITIVE_Z, NEGATIVE_X,
        NEGATIVE_X, NEGATIVE_Z, NEGATIVE_Z, POSITIVE_X,
        NEGATIVE_Y, NEGATIVE_Y
    }};

    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * 2 * 14, nullptr, GL_STATIC_DRAW);

    glBufferSubData(GL_ARRAY_BUFFER, 14 * sizeof(float) * 0, vertices.size() * sizeof(float) * 3, vertices.data());
    glBufferSubData(GL_ARRAY_BUFFER, 14 * sizeof(float) * 3, normals.size() * sizeof(float) * 3, normals.data());

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), reinterpret_cast<void*>(14 * sizeof(float) * 0));
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), reinterpret_cast<void*>(14 * sizeof(float) * 3));

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    glVertexAttribDivisor(0, 0);
    glVertexAttribDivisor(1, 0);

    //glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindBuffer(GL_ARRAY_BUFFER, m_attributes);
    glBufferData(GL_ARRAY_BUFFER, byteSize(), nullptr, GL_STATIC_DRAW);

    glBufferSubData(GL_ARRAY_BUFFER, verticesCount() * sizeof(float) * 0, verticesCount() * sizeof(float) * 3, m_position.data());
    glBufferSubData(GL_ARRAY_BUFFER, verticesCount() * sizeof(float) * 3, verticesCount() * sizeof(float) * 3, m_scale.data());
    glBufferSubData(GL_ARRAY_BUFFER, verticesCount() * sizeof(float) * 6, verticesCount() * sizeof(float) * 1, m_colorValue.data());

    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), reinterpret_cast<void*>(verticesCount() * sizeof(float) * 0));
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), reinterpret_cast<void*>(verticesCount() * sizeof(float) * 3));
    glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, sizeof(float), reinterpret_cast<void*>(verticesCount() * sizeof(float) * 6));

    glVertexAttribDivisor(2, 1);
    glVertexAttribDivisor(3, 1);
    glVertexAttribDivisor(4, 1);

    glEnableVertexAttribArray(2);
    glEnableVertexAttribArray(3);
    glEnableVertexAttribArray(4);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

bool CuboidInstancing::loadShader()
{
    const auto vertexShaderSource = loadShaderSource("/visualization-instancing/standard.vert");
    const auto vertexShaderSource_ptr = vertexShaderSource.c_str();
    if(vertexShaderSource_ptr)
        glShaderSource(m_vertexShader, 1, &vertexShaderSource_ptr, 0);

    glCompileShader(m_vertexShader);

    bool success = checkForCompilationError(m_vertexShader, "vertex shader");

    const auto fragmentShaderSource = loadShaderSource("/visualization.frag");
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

void CuboidInstancing::setCube(size_t index, const Cuboid & cuboid)
{
    m_position[index] = cuboid.center;
    m_scale[index] = cuboid.extent / glm::vec3(2.0f);
    m_colorValue[index] = cuboid.colorValue;
}

size_t CuboidInstancing::size() const
{
    return m_position.size();
}

size_t CuboidInstancing::verticesPerCuboid() const
{
    return 1;
}

size_t CuboidInstancing::verticesCount() const
{
    return size() * verticesPerCuboid();
}

size_t CuboidInstancing::staticByteSize() const
{
    return sizeof(glm::vec3) * 14 * 2;
}

size_t CuboidInstancing::byteSize() const
{
    return verticesCount() * vertexByteSize();
}

size_t CuboidInstancing::vertexByteSize() const
{
    return sizeof(float) * componentCount();
}

size_t CuboidInstancing::componentCount() const
{
    return 7;
}

void CuboidInstancing::resize(size_t count)
{
    m_position.resize(count * verticesPerCuboid());
    m_scale.resize(count * verticesPerCuboid());
    m_colorValue.resize(count * verticesPerCuboid());
}

void CuboidInstancing::onRender()
{
    glBindVertexArray(m_vao);

    glEnable(GL_DEPTH_TEST);
    //glDepthFunc(GL_LESS);

    glDisable(GL_CULL_FACE);
    //glCullFace(GL_BACK);

    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDepthMask(GL_TRUE);

    glUseProgram(m_program);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 14, size());

    glUseProgram(0);

    glBindVertexArray(0);
}

gl::GLuint CuboidInstancing::program() const
{
    return m_program;
}
