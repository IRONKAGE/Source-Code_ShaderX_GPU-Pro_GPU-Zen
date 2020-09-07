
#include "CuboidTriangleStrip.h"

#include <algorithm>

#include <glbinding/gl/gl.h>

#include "common.h"

using namespace gl;

CuboidTriangleStrip::CuboidTriangleStrip()
: CuboidImplementation("Triangle Strip")
, m_vertices(0)
, m_vao(0)
, m_vertexShader(0)
, m_fragmentShader(0)
{
}

CuboidTriangleStrip::~CuboidTriangleStrip()
{
    glDeleteBuffers(1, &m_vertices);
    glDeleteVertexArrays(1, &m_vao);
    glDeleteShader(m_vertexShader);
    glDeleteShader(m_fragmentShader);
    glDeleteProgram(m_program);
}

void CuboidTriangleStrip::onInitialize()
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

void CuboidTriangleStrip::initializeVAO()
{
    glBindVertexArray(m_vao);

    glBindBuffer(GL_ARRAY_BUFFER, m_vertices);
    glBufferData(GL_ARRAY_BUFFER, byteSize(), nullptr, GL_STATIC_DRAW);

    glBufferSubData(GL_ARRAY_BUFFER, verticesCount() * sizeof(float) * 0, verticesCount() * sizeof(float) * 3, m_vertex.data());
    glBufferSubData(GL_ARRAY_BUFFER, verticesCount() * sizeof(float) * 3, verticesCount() * sizeof(float) * 3, m_normal.data());
    glBufferSubData(GL_ARRAY_BUFFER, verticesCount() * sizeof(float) * 6, verticesCount() * sizeof(float) * 1, m_colorValue.data());

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), reinterpret_cast<void*>(verticesCount() * sizeof(float) * 0));
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), reinterpret_cast<void*>(verticesCount() * sizeof(float) * 3));
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(float), reinterpret_cast<void*>(verticesCount() * sizeof(float) * 6));

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

bool CuboidTriangleStrip::loadShader()
{
    const auto vertexShaderSource = loadShaderSource("/visualization-triangles/standard.vert");
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

void CuboidTriangleStrip::setCube(size_t index, const Cuboid & cuboid)
{
    static const auto NEGATIVE_X = glm::vec3(-1.0, 0.0, 0.0);
    static const auto NEGATIVE_Y = glm::vec3(0.0, -1.0, 0.0);
    static const auto NEGATIVE_Z = glm::vec3(0.0, 0.0, -1.0);
    static const auto POSITIVE_X = glm::vec3(1.0, 0.0, 0.0);
    static const auto POSITIVE_Y = glm::vec3(0.0, 1.0, 0.0);
    static const auto POSITIVE_Z = glm::vec3(0.0, 0.0, 1.0);

    const auto llf = cuboid.center - cuboid.extent / glm::vec3(2.0);
    const auto urb = cuboid.center + cuboid.extent / glm::vec3(2.0);

    const auto vertices = std::array<glm::vec3, 8>{{
        glm::vec3(llf.x, urb.y, llf.z), // A = H
        glm::vec3(llf.x, urb.y, urb.z), // B = F
        glm::vec3(urb.x, urb.y, llf.z), // C = J
        glm::vec3(urb.x, urb.y, urb.z), // D
        glm::vec3(urb.x, llf.y, urb.z), // E = L
        glm::vec3(llf.x, llf.y, urb.z), // G
        glm::vec3(llf.x, llf.y, llf.z), // I
        glm::vec3(urb.x, llf.y, llf.z) // K
    }};

    size_t i = 0;
    const auto emitVertex = [this, index, &cuboid, &i](const glm::vec3 & vertex, const glm::vec3 & normal)
    {
        m_vertex[index * verticesPerCuboid() + i] = vertex;
        m_normal[index * verticesPerCuboid() + i] = normal;
        m_colorValue[index * verticesPerCuboid() + i] = cuboid.colorValue;

        ++i;
    };

    emitVertex(vertices[0], POSITIVE_Y); // A
    emitVertex(vertices[1], POSITIVE_Y); // B
    emitVertex(vertices[2], POSITIVE_Y); // C
    emitVertex(vertices[3], POSITIVE_Y); // D

    emitVertex(vertices[4], POSITIVE_X); // E

    emitVertex(vertices[1], POSITIVE_Z); // F
    emitVertex(vertices[5], POSITIVE_Z); // G

    emitVertex(vertices[0], NEGATIVE_X); // H
    emitVertex(vertices[6], NEGATIVE_X); // I

    emitVertex(vertices[2], NEGATIVE_Z); // J
    emitVertex(vertices[7], NEGATIVE_Z); // K

    emitVertex(vertices[4], POSITIVE_X); // L

    emitVertex(vertices[6], NEGATIVE_Y); // I
    emitVertex(vertices[5], NEGATIVE_Y); // G
}

size_t CuboidTriangleStrip::size() const
{
    return m_vertex.size() / verticesPerCuboid();
}

size_t CuboidTriangleStrip::verticesPerCuboid() const
{
    return 14;
}

size_t CuboidTriangleStrip::verticesCount() const
{
    return size() * verticesPerCuboid();
}

size_t CuboidTriangleStrip::staticByteSize() const
{
    return 0;
}

size_t CuboidTriangleStrip::byteSize() const
{
    return verticesCount() * vertexByteSize();
}

size_t CuboidTriangleStrip::vertexByteSize() const
{
    return sizeof(float) * componentCount();
}

size_t CuboidTriangleStrip::componentCount() const
{
    return 7;
}

void CuboidTriangleStrip::resize(size_t count)
{
    m_vertex.resize(count * verticesPerCuboid());
    m_normal.resize(count * verticesPerCuboid());
    m_colorValue.resize(count * verticesPerCuboid());

    m_multiStarts.resize(count);
    m_multiCounts.resize(count);

    size_t next = 0;
    std::fill(m_multiCounts.begin(), m_multiCounts.end(), verticesPerCuboid());
    std::generate(m_multiStarts.begin(), m_multiStarts.end(), [this, &next]() {
        const auto current = next;
        next += verticesPerCuboid();
        return current;
    });
}

void CuboidTriangleStrip::onRender()
{
    glBindVertexArray(m_vao);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    glDisable(GL_CULL_FACE);
    //glCullFace(GL_BACK);

    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDepthMask(GL_TRUE);

    glUseProgram(m_program);
    glMultiDrawArrays(GL_TRIANGLE_STRIP, m_multiStarts.data(), m_multiCounts.data(), size());

    glDepthMask(GL_TRUE);

    glUseProgram(0);

    glBindVertexArray(0);
}

gl::GLuint CuboidTriangleStrip::program() const
{
    return m_program;
}
