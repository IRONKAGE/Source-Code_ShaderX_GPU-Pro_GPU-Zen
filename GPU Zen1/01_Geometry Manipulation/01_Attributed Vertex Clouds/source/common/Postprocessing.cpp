
#include "Postprocessing.h"

#include <glbinding/gl/gl.h>

#include "common.h"

using namespace gl;

Postprocessing::Postprocessing()
: m_initialized(false)
, m_fbo(0)
, m_colorBuffer(0)
, m_depthBuffer(0)
, m_postprocessingVertices(0)
, m_postprocessingVAO(0)
, m_postProcessingProgram(0)
, m_postProcessingVertexShader(0)
, m_postProcessingFragmentShader(0)
{
}

Postprocessing::~Postprocessing()
{
    glDeleteFramebuffers(1, &m_fbo);
    glDeleteTextures(1, &m_colorBuffer);
    glDeleteTextures(1, &m_depthBuffer);
    glDeleteBuffers(1, &m_postprocessingVertices);
    glDeleteVertexArrays(1, &m_postprocessingVAO);
    glDeleteShader(m_postProcessingVertexShader);
    glDeleteShader(m_postProcessingFragmentShader);
    glDeleteProgram(m_postProcessingProgram);
}

bool Postprocessing::initialized() const
{
    return m_initialized;
}

void Postprocessing::initialize()
{
    if (!initialized())
    {
        onInitialize();

        m_initialized = true;
    }
}

void Postprocessing::render()
{
    initialize();

    onRender();
}

void Postprocessing::onInitialize()
{
    glGenFramebuffers(1, &m_fbo);
    glGenTextures(1, &m_colorBuffer);
    glGenTextures(1, &m_depthBuffer);
    glGenBuffers(1, &m_postprocessingVertices);
    glGenVertexArrays(1, &m_postprocessingVAO);
    m_postProcessingVertexShader = glCreateShader(GL_VERTEX_SHADER);
    m_postProcessingFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    m_postProcessingProgram = glCreateProgram();

    glBindTexture(GL_TEXTURE_2D, m_colorBuffer);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, m_width, m_height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    //glBindTexture(GL_TEXTURE_2D, 0);

    glBindTexture(GL_TEXTURE_2D, m_depthBuffer);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    //glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, m_width, m_height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, m_colorBuffer, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, m_depthBuffer, 0);

    std::array<glm::vec2, 3> sat = {{
        glm::vec2(1.0f, -1.0f),
        glm::vec2(1.0f, 3.0f),
        glm::vec2(-3.0f, -1.0f)
    }};

    glBindVertexArray(m_postprocessingVAO);

    glBindBuffer(GL_ARRAY_BUFFER, m_postprocessingVertices);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * sat.size(), sat.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), nullptr);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glAttachShader(m_postProcessingProgram, m_postProcessingVertexShader);
    glAttachShader(m_postProcessingProgram, m_postProcessingFragmentShader);

    loadShader();
}

bool Postprocessing::loadShader()
{
    const auto vertexShaderSource = loadShaderSource("/postprocessing/standard.vert");
    const auto vertexShaderSource_ptr = vertexShaderSource.c_str();
    if(vertexShaderSource_ptr)
        glShaderSource(m_postProcessingVertexShader, 1, &vertexShaderSource_ptr, 0);

    glCompileShader(m_postProcessingVertexShader);

    bool success = checkForCompilationError(m_postProcessingVertexShader, "postprocessing vertex shader");

    const auto fragmentShaderSource = loadShaderSource("/postprocessing/standard.frag");
    const auto fragmentShaderSource_ptr = fragmentShaderSource.c_str();
    if(fragmentShaderSource_ptr)
        glShaderSource(m_postProcessingFragmentShader, 1, &fragmentShaderSource_ptr, 0);

    glCompileShader(m_postProcessingFragmentShader);

    success &= checkForCompilationError(m_postProcessingFragmentShader, "postprocessing fragment shader");


    if (!success)
    {
        return false;
    }

    glLinkProgram(m_postProcessingProgram);

    success &= checkForLinkerError(m_postProcessingProgram, "postprocessing program");

    if (!success)
    {
        return false;
    }

    glUseProgram(m_postProcessingProgram);
    glUniform1i(glGetUniformLocation(m_postProcessingProgram, "colorBuffer"), 0);
    glUniform1i(glGetUniformLocation(m_postProcessingProgram, "depthBuffer"), 1);
    glUseProgram(0);

    glBindFragDataLocation(m_postProcessingProgram, 0, "out_color");

    return true;
}

void Postprocessing::onRender()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_colorBuffer);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, m_depthBuffer);

    glBindVertexArray(m_postprocessingVAO);

    glUseProgram(m_postProcessingProgram);

    glDrawArrays(GL_TRIANGLES, 0, 3);

    glUseProgram(0);

    glBindVertexArray(0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);

    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
}

gl::GLuint Postprocessing::fbo() const
{
    return m_fbo;
}

void Postprocessing::resize(int w, int h)
{
    if (initialized())
    {
        onResize(w, h);
    }
}

void Postprocessing::onResize(int w, int h)
{
    glBindTexture(GL_TEXTURE_2D, m_colorBuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    //glBindTexture(GL_TEXTURE_2D, 0);

    glBindTexture(GL_TEXTURE_2D, m_depthBuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, w, h, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);
}
