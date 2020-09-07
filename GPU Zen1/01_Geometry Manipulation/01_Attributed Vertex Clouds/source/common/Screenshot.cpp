
#include "Screenshot.h"

#include <fstream>
#include <iostream>

#include <glbinding/gl/gl.h>

#include "common.h"

#if defined(_WIN32)
#include <direct.h>
#endif

#include <sys/stat.h>

using namespace gl;

Screenshot::Screenshot()
: m_initialized(false)
, m_width(0)
, m_height(0)
, m_fbo(0)
, m_colorBuffer(0)
, m_depthBuffer(0)
{
}

Screenshot::~Screenshot()
{
    glDeleteFramebuffers(1, &m_fbo);
    glDeleteTextures(1, &m_colorBuffer);
    glDeleteTextures(1, &m_depthBuffer);
}

bool Screenshot::initialized() const
{
    return m_initialized;
}

void Screenshot::initialize()
{
    if (!initialized())
    {
        onInitialize();

        m_initialized = true;
    }
}

void Screenshot::onInitialize()
{
    glGenFramebuffers(1, &m_fbo);
    glGenTextures(1, &m_colorBuffer);
    glGenTextures(1, &m_depthBuffer);

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
}

gl::GLuint Screenshot::fbo() const
{
    return m_fbo;
}

void Screenshot::resize(int w, int h)
{
    if (initialized())
    {
        onResize(w, h);
    }
}

void Screenshot::onResize(int w, int h)
{
    m_width = w;
    m_height = h;

    glBindTexture(GL_TEXTURE_2D, m_colorBuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    //glBindTexture(GL_TEXTURE_2D, 0);

    glBindTexture(GL_TEXTURE_2D, m_depthBuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, w, h, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void Screenshot::saveScreenshot(const std::string & filename)
{
#if defined(_WIN32)
    /*int ret = */_mkdir("screenshots");
#else
    /*int ret = */mkdir("screenshots", 0755);
#endif

    std::vector<char> data(m_width * m_height * 3 * sizeof(char));

    glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
    glReadPixels(0, 0, m_width, m_height, GL_RGB, GL_UNSIGNED_BYTE, data.data());
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    auto ppm = std::ofstream();
    ppm.open("screenshots/"+filename, std::ios::binary);

    if (ppm.fail())
    {
        std::cerr << "Cannot open output file 'screenshots/" << filename << "'." << std::endl;
        ppm.close();
        return;
    }

    ppm << "P6\n" << m_width << " " << m_height << "\n255\n"; // PPM header

    // mirror the image horizontally
    const auto scanLineSize = m_width * 3;
    for(auto y = static_cast<int>(m_height) - 1; y >= 0; --y)
        ppm.write(reinterpret_cast<const char *>(data.data()) + scanLineSize * y, scanLineSize);

    ppm.close();
}
