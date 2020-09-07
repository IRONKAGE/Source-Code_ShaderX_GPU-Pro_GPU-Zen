
#pragma once

#include <vector>

#include <glbinding/gl/types.h>


class Screenshot
{
public:
    Screenshot();
    ~Screenshot();

    void saveScreenshot(const std::string & filename);

    bool initialized() const;

    void initialize();
    void resize(int w, int h);

    void onInitialize();
    void onRender();
    void onResize(int w, int h);

    gl::GLuint fbo() const;
public:
    bool m_initialized;
    int m_width;
    int m_height;

    gl::GLuint m_fbo;
    gl::GLuint m_colorBuffer;
    gl::GLuint m_depthBuffer;
};
