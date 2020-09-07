
#include "BlockWorldVertexCloud.h"

#include <glbinding/gl/gl.h>

#include "common.h"

using namespace gl;

BlockWorldVertexCloud::BlockWorldVertexCloud()
: BlockWorldImplementation("Attributed Vertex Cloud")
, m_vertices(0)
, m_vao(0)
, m_vertexShader(0)
, m_geometryShader(0)
, m_fragmentShader(0)
{
}

BlockWorldVertexCloud::~BlockWorldVertexCloud()
{
    glDeleteBuffers(1, &m_vertices);
    glDeleteVertexArrays(1, &m_vao);
    glDeleteShader(m_vertexShader);
    glDeleteShader(m_geometryShader);
    glDeleteShader(m_fragmentShader);
    glDeleteProgram(m_program);
}

void BlockWorldVertexCloud::onInitialize()
{
    glGenBuffers(1, &m_vertices);
    glGenVertexArrays(1, &m_vao);

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

void BlockWorldVertexCloud::initializeVAO()
{
    glBindVertexArray(m_vao);

    glBindBuffer(GL_ARRAY_BUFFER, m_vertices);
    glBufferData(GL_ARRAY_BUFFER, byteSize(), nullptr, GL_STATIC_DRAW);

    glBufferSubData(GL_ARRAY_BUFFER, verticesCount() * sizeof(float) * 0, verticesCount() * sizeof(float) * 4, m_positionAndType.data());

    glVertexAttribIPointer(0, 4, GL_INT, sizeof(glm::ivec4), reinterpret_cast<void*>(verticesCount() * sizeof(float) * 0));

    glEnableVertexAttribArray(0);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

bool BlockWorldVertexCloud::loadShader()
{
    const auto vertexShaderSource = loadShaderSource("/blockworld-avc/standard.vert");
    const auto vertexShaderSource_ptr = vertexShaderSource.c_str();
    if(vertexShaderSource_ptr)
        glShaderSource(m_vertexShader, 1, &vertexShaderSource_ptr, 0);

    glCompileShader(m_vertexShader);

    bool success = checkForCompilationError(m_vertexShader, "vertex shader");


    const auto geometryShaderSource = loadShaderSource("/blockworld-avc/standard.geom");
    const auto geometryShaderSource_ptr = geometryShaderSource.c_str();
    if(geometryShaderSource_ptr)
        glShaderSource(m_geometryShader, 1, &geometryShaderSource_ptr, 0);

    glCompileShader(m_geometryShader);

    success &= checkForCompilationError(m_geometryShader, "geometry shader");


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

    glUseProgram(m_program);
    glUniform1f(glGetUniformLocation(m_program, "blockSize"), m_blockSize);
    glUseProgram(0);

    glBindFragDataLocation(m_program, 0, "out_color");

    return true;
}

void BlockWorldVertexCloud::setBlock(size_t index, const Block & block)
{
    m_positionAndType[index] = glm::ivec4(block.position, block.type);
}

size_t BlockWorldVertexCloud::size() const
{
    return m_positionAndType.size();
}

size_t BlockWorldVertexCloud::verticesPerCuboid() const
{
    return 1;
}

size_t BlockWorldVertexCloud::verticesCount() const
{
    return size() * verticesPerCuboid();
}

size_t BlockWorldVertexCloud::staticByteSize() const
{
    return 0;
}

size_t BlockWorldVertexCloud::byteSize() const
{
    return verticesPerCuboid() * size() * vertexByteSize();
}

size_t BlockWorldVertexCloud::vertexByteSize() const
{
    return sizeof(float) * componentCount();
}

size_t BlockWorldVertexCloud::componentCount() const
{
    return 4;
}

void BlockWorldVertexCloud::resize(size_t count)
{
    m_positionAndType.resize(count);
}

void BlockWorldVertexCloud::onRender()
{
    glBindVertexArray(m_vao);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDepthMask(GL_TRUE);

    glUseProgram(m_program);
    glDrawArrays(GL_POINTS, 0, size());

    glUseProgram(0);

    glBindVertexArray(0);
}

gl::GLuint BlockWorldVertexCloud::program() const
{
    return m_program;
}
