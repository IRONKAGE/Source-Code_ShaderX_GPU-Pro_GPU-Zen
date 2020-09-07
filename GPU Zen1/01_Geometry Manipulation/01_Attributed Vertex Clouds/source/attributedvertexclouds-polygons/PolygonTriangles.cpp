
#include "PolygonTriangles.h"

#include <glbinding/gl/gl.h>

#include "common.h"

using namespace gl;

PolygonTriangles::PolygonTriangles()
: PolygonImplementation("Triangles")
, m_vertices(0)
, m_vao(0)
, m_vertexShader(0)
, m_fragmentShader(0)
{
}

PolygonTriangles::~PolygonTriangles()
{
    glDeleteBuffers(1, &m_vertices);
    glDeleteVertexArrays(1, &m_vao);

    glDeleteShader(m_vertexShader);
    glDeleteShader(m_fragmentShader);
    glDeleteProgram(m_program);
}

void PolygonTriangles::onInitialize()
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

void PolygonTriangles::initializeVAO()
{
    glBindVertexArray(m_vao);

    glBindBuffer(GL_ARRAY_BUFFER, m_vertices);
    glBufferData(GL_ARRAY_BUFFER, size() * vertexByteSize(), nullptr, GL_STATIC_DRAW);

    glBufferSubData(GL_ARRAY_BUFFER, static_cast<gl::GLintptr>(size() * sizeof(float) * 0), size() * sizeof(float) * 3, m_position.data());
    glBufferSubData(GL_ARRAY_BUFFER, static_cast<gl::GLintptr>(size() * sizeof(float) * 3), size() * sizeof(float) * 3, m_normal.data());
    glBufferSubData(GL_ARRAY_BUFFER, static_cast<gl::GLintptr>(size() * sizeof(float) * 6), size() * sizeof(float) * 1, m_colorValue.data());

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), reinterpret_cast<void*>(size() * sizeof(float) * 0));
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), reinterpret_cast<void*>(size() * sizeof(float) * 3));
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(float), reinterpret_cast<void*>(size() * sizeof(float) * 6));

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

bool PolygonTriangles::loadShader()
{
    const auto vertexShaderSource = loadShaderSource("/visualization-triangles/standard.vert");
    const auto vertexShaderSource_ptr = vertexShaderSource.c_str();
    if(vertexShaderSource_ptr)
        glShaderSource(m_vertexShader, 1, &vertexShaderSource_ptr, nullptr);

    glCompileShader(m_vertexShader);

    bool success = checkForCompilationError(m_vertexShader, "vertex shader");


    const auto fragmentShaderSource = loadShaderSource("/visualization.frag");
    const auto fragmentShaderSource_ptr = fragmentShaderSource.c_str();
    if(fragmentShaderSource_ptr)
        glShaderSource(m_fragmentShader, 1, &fragmentShaderSource_ptr, nullptr);

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

void PolygonTriangles::setPolygon(size_t /*index*/, const Polygon & polygon)
{
    if (polygon.points.size() < 3)
    {
        return;
    }

    const auto vertexCount = (4 * polygon.points.size() - 4) * 3;

    m_mutex.lock();

    const auto firstIndex = m_position.size();
    const auto topFaceStartIndex = firstIndex + 2 * polygon.points.size() * 3;
    const auto bottomFaceStartIndex = topFaceStartIndex + (polygon.points.size() - 2) * 3;

    m_position.resize(m_position.size() + vertexCount);
    m_normal.resize(m_normal.size() + vertexCount);
    m_colorValue.resize(m_colorValue.size() + vertexCount);

    for (auto i = size_t(0); i < polygon.points.size(); ++i)
    {
        if (i >= 2)
        {
            // Top face
            m_position[topFaceStartIndex + 3*(i-2)+0] = glm::vec3(polygon.points[i-1].x, polygon.heightRange.y, polygon.points[i-1].y);
            m_position[topFaceStartIndex + 3*(i-2)+1] = glm::vec3(polygon.points[0].x, polygon.heightRange.y, polygon.points[0].y);
            m_position[topFaceStartIndex + 3*(i-2)+2] = glm::vec3(polygon.points[i].x, polygon.heightRange.y, polygon.points[i].y);
            m_normal[topFaceStartIndex + 3*(i-2)+0] = glm::vec3(0.0f, 1.0f, 0.0f);
            m_normal[topFaceStartIndex + 3*(i-2)+1] = glm::vec3(0.0f, 1.0f, 0.0f);
            m_normal[topFaceStartIndex + 3*(i-2)+2] = glm::vec3(0.0f, 1.0f, 0.0f);
            m_colorValue[topFaceStartIndex + 3*(i-2)+0] = polygon.colorValue;
            m_colorValue[topFaceStartIndex + 3*(i-2)+1] = polygon.colorValue;
            m_colorValue[topFaceStartIndex + 3*(i-2)+2] = polygon.colorValue;

            // Bottom face
            m_position[bottomFaceStartIndex + 3*(i-2)+0] = glm::vec3(polygon.points[i].x, polygon.heightRange.x, polygon.points[i].y);
            m_position[bottomFaceStartIndex + 3*(i-2)+1] = glm::vec3(polygon.points[0].x, polygon.heightRange.x, polygon.points[0].y);
            m_position[bottomFaceStartIndex + 3*(i-2)+2] = glm::vec3(polygon.points[i-1].x, polygon.heightRange.x, polygon.points[i-1].y);
            m_normal[bottomFaceStartIndex + 3*(i-2)+0] = glm::vec3(0.0f, -1.0f, 0.0f);
            m_normal[bottomFaceStartIndex + 3*(i-2)+1] = glm::vec3(0.0f, -1.0f, 0.0f);
            m_normal[bottomFaceStartIndex + 3*(i-2)+2] = glm::vec3(0.0f, -1.0f, 0.0f);
            m_colorValue[bottomFaceStartIndex + 3*(i-2)+0] = polygon.colorValue;
            m_colorValue[bottomFaceStartIndex + 3*(i-2)+1] = polygon.colorValue;
            m_colorValue[bottomFaceStartIndex + 3*(i-2)+2] = polygon.colorValue;
        }

        // Side face
        const auto & current = polygon.points[i];
        const auto & next = polygon.points[(i+1) % polygon.points.size()];

        const auto normal = glm::cross(glm::vec3(next.x - current.x, 0.0f, next.y - current.y), glm::vec3(0.0f, 1.0f, 0.0f));

        m_position[firstIndex + 6*i+0] = glm::vec3(next.x, polygon.heightRange.x, next.y);
        m_position[firstIndex + 6*i+1] = glm::vec3(current.x, polygon.heightRange.x, current.y);
        m_position[firstIndex + 6*i+2] = glm::vec3(next.x, polygon.heightRange.y, next.y);
        m_position[firstIndex + 6*i+3] = glm::vec3(next.x, polygon.heightRange.y, next.y);
        m_position[firstIndex + 6*i+4] = glm::vec3(current.x, polygon.heightRange.x, current.y);
        m_position[firstIndex + 6*i+5] = glm::vec3(current.x, polygon.heightRange.y, current.y);
        m_normal[firstIndex + 6*i+0] = normal;
        m_normal[firstIndex + 6*i+1] = normal;
        m_normal[firstIndex + 6*i+2] = normal;
        m_normal[firstIndex + 6*i+3] = normal;
        m_normal[firstIndex + 6*i+4] = normal;
        m_normal[firstIndex + 6*i+5] = normal;
        m_colorValue[firstIndex + 6*i+0] = polygon.colorValue;
        m_colorValue[firstIndex + 6*i+1] = polygon.colorValue;
        m_colorValue[firstIndex + 6*i+2] = polygon.colorValue;
        m_colorValue[firstIndex + 6*i+3] = polygon.colorValue;
        m_colorValue[firstIndex + 6*i+4] = polygon.colorValue;
        m_colorValue[firstIndex + 6*i+5] = polygon.colorValue;
    }

    m_mutex.unlock();
}

size_t PolygonTriangles::size() const
{
    return m_position.size();
}

size_t PolygonTriangles::verticesCount() const
{
    return size();
}

size_t PolygonTriangles::staticByteSize() const
{
    return 0;
}

size_t PolygonTriangles::byteSize() const
{
    return size() * vertexByteSize();
}

size_t PolygonTriangles::vertexByteSize() const
{
    return sizeof(float) * componentCount();
}

size_t PolygonTriangles::componentCount() const
{
    return 7;
}

void PolygonTriangles::resize(size_t /*count*/)
{
    // TODO: should implement
}

void PolygonTriangles::onRender()
{
    glBindVertexArray(m_vao);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDepthMask(GL_TRUE);

    glUseProgram(m_program);
    glDrawArrays(GL_TRIANGLES, 0, static_cast<gl::GLint>(verticesCount()));

    glUseProgram(0);

    glBindVertexArray(0);
}

gl::GLuint PolygonTriangles::program() const
{
    return m_program;
}
