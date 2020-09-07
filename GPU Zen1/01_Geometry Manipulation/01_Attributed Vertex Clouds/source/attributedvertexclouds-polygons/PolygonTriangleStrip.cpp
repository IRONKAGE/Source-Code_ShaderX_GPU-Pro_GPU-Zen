
#include "PolygonTriangleStrip.h"

#include <glbinding/gl/gl.h>

#include "common.h"

using namespace gl;

PolygonTriangleStrip::PolygonTriangleStrip()
: PolygonImplementation("Triangle Strip")
, m_vertices(0)
, m_vao(0)
, m_vertexShader(0)
, m_fragmentShader(0)
{
}

PolygonTriangleStrip::~PolygonTriangleStrip()
{
    glDeleteBuffers(1, &m_vertices);
    glDeleteVertexArrays(1, &m_vao);

    glDeleteShader(m_vertexShader);
    glDeleteShader(m_fragmentShader);
    glDeleteProgram(m_program);
}

void PolygonTriangleStrip::onInitialize()
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

void PolygonTriangleStrip::initializeVAO()
{
    glBindVertexArray(m_vao);

    glBindBuffer(GL_ARRAY_BUFFER, m_vertices);
    glBufferData(GL_ARRAY_BUFFER, size() * vertexByteSize(), nullptr, GL_STATIC_DRAW);

    glBufferSubData(GL_ARRAY_BUFFER, size() * sizeof(float) * 0, size() * sizeof(float) * 3, m_position.data());
    glBufferSubData(GL_ARRAY_BUFFER, size() * sizeof(float) * 3, size() * sizeof(float) * 3, m_normal.data());
    glBufferSubData(GL_ARRAY_BUFFER, size() * sizeof(float) * 6, size() * sizeof(float) * 1, m_colorValue.data());

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), reinterpret_cast<void*>(size() * sizeof(float) * 0));
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), reinterpret_cast<void*>(size() * sizeof(float) * 3));
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(float), reinterpret_cast<void*>(size() * sizeof(float) * 6));

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

bool PolygonTriangleStrip::loadShader()
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

void PolygonTriangleStrip::setPolygon(size_t index, const Polygon & polygon)
{
    if (polygon.points.size() < 3)
    {
        return;
    }

    const auto vertexCount = 2 + 4 * polygon.points.size();

    m_mutex.lock();

    const auto firstIndex = m_position.size();
    const auto topFaceStartIndex = firstIndex + 2 + 2 * polygon.points.size();
    const auto bottomFaceStartIndex = topFaceStartIndex + polygon.points.size();

    m_multiStarts.at(3 * index + 0) = firstIndex;
    m_multiStarts.at(3 * index + 1) = topFaceStartIndex;
    m_multiStarts.at(3 * index + 2) = bottomFaceStartIndex;

    m_multiCounts.at(3 * index + 0) = 2 + 2 * polygon.points.size();
    m_multiCounts.at(3 * index + 1) = polygon.points.size();
    m_multiCounts.at(3 * index + 2) = polygon.points.size();

    m_position.resize(m_position.size() + vertexCount);
    m_normal.resize(m_normal.size() + vertexCount);
    m_colorValue.resize(m_colorValue.size() + vertexCount);

    // Side faces
    for (auto i = size_t(0); i <= polygon.points.size(); ++i)
    {
        const auto & current = polygon.points[i % polygon.points.size()];
        const auto & previous = polygon.points[(i + polygon.points.size() - 1) % polygon.points.size()];

        const auto normal = glm::cross(glm::vec3(current.x - previous.x, 0.0f, current.y - previous.y), glm::vec3(0.0f, 1.0f, 0.0f));

        m_position[firstIndex + 2*i+0] = glm::vec3(current.x, polygon.heightRange.x, current.y);
        m_position[firstIndex + 2*i+1] = glm::vec3(current.x, polygon.heightRange.y, current.y);
        m_normal[firstIndex + 2*i+0] = normal;
        m_normal[firstIndex + 2*i+1] = normal;
        m_colorValue[firstIndex + 2*i+0] = polygon.colorValue;
        m_colorValue[firstIndex + 2*i+1] = polygon.colorValue;
    }

    // Top face
    for (auto i = size_t(0); i < polygon.points.size(); ++i)
    {
        if (i > 0)
        {
            auto j = i / 2;
            if (i % 2)
            {
                m_position[topFaceStartIndex + i] = glm::vec3(polygon.points[polygon.points.size() - j - 1].x, polygon.heightRange.y, polygon.points[polygon.points.size() - j - 1].y);
            }
            else
            {
                m_position[topFaceStartIndex + i] = glm::vec3(polygon.points[j].x, polygon.heightRange.y, polygon.points[j].y);
            }
        }
        else
        {
            m_position[topFaceStartIndex + i] = glm::vec3(polygon.points[i].x, polygon.heightRange.y, polygon.points[i].y);
        }

        m_normal[topFaceStartIndex + i] = glm::vec3(0.0f, 1.0f, 0.0f);
        m_colorValue[topFaceStartIndex + i] = polygon.colorValue;
    }

    // Bottom face
    for (auto i = size_t(0); i < polygon.points.size(); ++i)
    {
        if (i > 0)
        {
            auto j = i / 2;
            if (i % 2)
            {
                m_position[bottomFaceStartIndex + i] = glm::vec3(polygon.points[j].x, polygon.heightRange.x, polygon.points[j].y);
            }
            else
            {
                m_position[bottomFaceStartIndex + i] = glm::vec3(polygon.points[polygon.points.size() - j - 1].x, polygon.heightRange.x, polygon.points[polygon.points.size() - j - 1].y);
            }
        }
        else
        {
            m_position[bottomFaceStartIndex + i] = glm::vec3(polygon.points[i].x, polygon.heightRange.x, polygon.points[i].y);
        }

        m_normal[bottomFaceStartIndex + i] = glm::vec3(0.0f, -1.0f, 0.0f);
        m_colorValue[bottomFaceStartIndex + i] = polygon.colorValue;
    }

    m_mutex.unlock();
}

size_t PolygonTriangleStrip::size() const
{
    return m_position.size();
}

size_t PolygonTriangleStrip::verticesCount() const
{
    return size();
}

size_t PolygonTriangleStrip::staticByteSize() const
{
    return 0;
}

size_t PolygonTriangleStrip::byteSize() const
{
    return size() * vertexByteSize();
}

size_t PolygonTriangleStrip::vertexByteSize() const
{
    return sizeof(float) * componentCount();
}

size_t PolygonTriangleStrip::componentCount() const
{
    return 7;
}

void PolygonTriangleStrip::resize(size_t count)
{
    m_multiStarts.resize(3 * count);
    m_multiCounts.resize(3 * count);
}

void PolygonTriangleStrip::onRender()
{
    glBindVertexArray(m_vao);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDepthMask(GL_TRUE);

    glUseProgram(m_program);
    glMultiDrawArrays(GL_TRIANGLE_STRIP, m_multiStarts.data(), m_multiCounts.data(), m_multiStarts.size());

    glUseProgram(0);

    glBindVertexArray(0);
}

gl::GLuint PolygonTriangleStrip::program() const
{
    return m_program;
}
