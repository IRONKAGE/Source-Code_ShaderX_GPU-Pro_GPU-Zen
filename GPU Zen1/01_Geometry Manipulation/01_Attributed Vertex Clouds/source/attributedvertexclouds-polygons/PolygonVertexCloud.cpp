
#include "PolygonVertexCloud.h"

#include <glbinding/gl/gl.h>

#include "common.h"

using namespace gl;

PolygonVertexCloud::PolygonVertexCloud()
: PolygonImplementation("Attributed Vertex Cloud")
, m_vertices(0)
, m_centerHeightRangeBuffer(0)
, m_colorValueBuffer(0)
, m_centerHeightRangeTexture(0)
, m_colorValueTexture(0)
, m_vao(0)
, m_vertexShader(0)
, m_geometryShader(0)
, m_fragmentShader(0)
{
}

PolygonVertexCloud::~PolygonVertexCloud()
{
    glDeleteBuffers(1, &m_vertices);
    glDeleteBuffers(1, &m_centerHeightRangeBuffer);
    glDeleteBuffers(1, &m_colorValueBuffer);

    glDeleteVertexArrays(1, &m_vao);

    glDeleteTextures(1, &m_centerHeightRangeTexture);
    glDeleteTextures(1, &m_colorValueTexture);

    glDeleteShader(m_vertexShader);
    glDeleteShader(m_geometryShader);
    glDeleteShader(m_fragmentShader);
    glDeleteProgram(m_program);
}

void PolygonVertexCloud::onInitialize()
{
    glGenBuffers(1, &m_vertices);
    glGenBuffers(1, &m_centerHeightRangeBuffer);
    glGenBuffers(1, &m_colorValueBuffer);

    glGenVertexArrays(1, &m_vao);

    glGenTextures(1, &m_centerHeightRangeTexture);
    glGenTextures(1, &m_colorValueTexture);

    initializeVAO();

    m_vertexShader = glCreateShader(GL_VERTEX_SHADER);
    m_geometryShader = glCreateShader(GL_GEOMETRY_SHADER);
    m_fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

    m_program = glCreateProgram();

    glAttachShader(m_program, m_vertexShader);
    glAttachShader(m_program, m_geometryShader);
    glAttachShader(m_program, m_fragmentShader);

    loadShader();
}

void PolygonVertexCloud::initializeVAO()
{
    glBindVertexArray(m_vao);

    glBindBuffer(GL_ARRAY_BUFFER, m_vertices);
    glBufferData(GL_ARRAY_BUFFER, size() * vertexByteSize(), nullptr, GL_STATIC_DRAW);

    glBufferSubData(GL_ARRAY_BUFFER, size() * sizeof(float) * 0, size() * sizeof(float) * 2, m_positions.data());
    glBufferSubData(GL_ARRAY_BUFFER, size() * sizeof(float) * 2, size() * sizeof(float) * 1, m_polygonIndices.data());

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), reinterpret_cast<void*>(size() * sizeof(float) * 0));
    glVertexAttribIPointer(1, 1, GL_INT, sizeof(int), reinterpret_cast<void*>(size() * sizeof(float) * 2));
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), reinterpret_cast<void*>(size() * sizeof(float) * 0 + sizeof(float) * 2));
    glVertexAttribIPointer(3, 1, GL_INT, sizeof(int), reinterpret_cast<void*>(size() * sizeof(float) * 2 + sizeof(float)));

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glEnableVertexAttribArray(3);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);


    glBindBuffer(GL_ARRAY_BUFFER, m_centerHeightRangeBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 4 * m_center.size(), nullptr, GL_STATIC_DRAW);

    for (auto i = size_t(0); i < m_center.size(); ++i)
    {
        glBufferSubData(GL_ARRAY_BUFFER, sizeof(float) * 4 * i, 2 * sizeof(float), &m_center[i]);
        glBufferSubData(GL_ARRAY_BUFFER, sizeof(float) * (4 * i + 2), 2 * sizeof(float), &m_heightRange[i]);
    }

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindTexture(GL_TEXTURE_BUFFER, m_centerHeightRangeTexture);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, m_centerHeightRangeBuffer);

    glBindBuffer(GL_TEXTURE_BUFFER, m_colorValueBuffer);
    glBufferData(GL_TEXTURE_BUFFER, sizeof(float) * 1 * m_center.size(), nullptr, GL_STATIC_DRAW);

    glBufferSubData(GL_TEXTURE_BUFFER, 0, sizeof(float) * 1 * m_colorValue.size(), m_colorValue.data());

    glBindTexture(GL_TEXTURE_BUFFER, m_colorValueTexture);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, m_colorValueBuffer);

    glBindBuffer(GL_TEXTURE_BUFFER, 0);
    glBindTexture(GL_TEXTURE_BUFFER, 0);
}

bool PolygonVertexCloud::loadShader()
{
    const auto vertexShaderSource = loadShaderSource("/polygons-avc/standard.vert");
    const auto vertexShaderSource_ptr = vertexShaderSource.c_str();
    if(vertexShaderSource_ptr)
        glShaderSource(m_vertexShader, 1, &vertexShaderSource_ptr, 0);

    glCompileShader(m_vertexShader);

    bool success = checkForCompilationError(m_vertexShader, "vertex shader");


    const auto geometryShaderSource = loadShaderSource("/polygons-avc/standard.geom");
    const auto geometryShaderSource_ptr = geometryShaderSource.c_str();
    if(geometryShaderSource_ptr)
        glShaderSource(m_geometryShader, 1, &geometryShaderSource_ptr, 0);

    glCompileShader(m_geometryShader);

    success &= checkForCompilationError(m_geometryShader, "geometry shader");


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

void PolygonVertexCloud::setPolygon(size_t index, const Polygon & polygon)
{
    m_center[index] = glm::vec2(0.0f, 0.0f);
    m_heightRange[index] = polygon.heightRange;
    m_colorValue[index] = polygon.colorValue;

    if (polygon.points.empty())
    {
        return;
    }

    m_mutex.lock();

    const auto firstIndex = m_positions.size();
    m_positions.resize(m_positions.size() + polygon.points.size() + 1);
    m_polygonIndices.resize(m_polygonIndices.size() + polygon.points.size() + 1);

    for (auto i = size_t(0); i < polygon.points.size(); ++i)
    {
        m_positions.at(firstIndex + i) = polygon.points.at(i);
        m_polygonIndices.at(firstIndex + i) = index;

        m_center[index] += polygon.points.at(i);
    }

    m_positions.at(firstIndex + polygon.points.size()) = polygon.points.front();
    m_polygonIndices.at(firstIndex + polygon.points.size()) = index;

    m_mutex.unlock();

    m_center[index] /= glm::vec2(polygon.points.size());
}

size_t PolygonVertexCloud::size() const
{
    return m_positions.size();
}

size_t PolygonVertexCloud::verticesCount() const
{
    return size();
}

size_t PolygonVertexCloud::staticByteSize() const
{
    return sizeof(float) * 6 * m_center.size();
}

size_t PolygonVertexCloud::byteSize() const
{
    return size() * vertexByteSize();
}

size_t PolygonVertexCloud::vertexByteSize() const
{
    return sizeof(float) * componentCount();
}

size_t PolygonVertexCloud::componentCount() const
{
    return 3;
}

void PolygonVertexCloud::resize(size_t count)
{
    m_center.resize(count);
    m_heightRange.resize(count);
    m_colorValue.resize(count);
}

void PolygonVertexCloud::onRender()
{
    glBindVertexArray(m_vao);


    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_BUFFER, m_centerHeightRangeTexture);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_BUFFER, m_colorValueTexture);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDepthMask(GL_TRUE);

    glUseProgram(m_program);
    const auto centerAndHeightsLocation = glGetUniformLocation(m_program, "centerAndHeights");
    const auto colorValuesLocation = glGetUniformLocation(m_program, "colorValues");
    glUniform1i(centerAndHeightsLocation, 1);
    glUniform1i(colorValuesLocation, 2);
    glDrawArrays(GL_POINTS, 0, size()-1);

    glUseProgram(0);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_BUFFER, 0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_BUFFER, 0);

    glBindVertexArray(0);
}

gl::GLuint PolygonVertexCloud::program() const
{
    return m_program;
}
