
#pragma once

#include <vector>

#include <glbinding/gl/types.h>


class Postprocessing
{
public:
    Postprocessing();
    ~Postprocessing();

    bool initialized() const;

    void initialize();
    void render();
    void resize(int w, int h);

    void onInitialize();
    void onRender();
    void onResize(int w, int h);

    bool loadShader();

    gl::GLuint fbo() const;
public:
    bool m_initialized;

    gl::GLuint m_fbo;
    gl::GLuint m_colorBuffer;
    gl::GLuint m_depthBuffer;

    gl::GLuint m_postprocessingVertices;
    gl::GLuint m_postprocessingVAO;
    gl::GLuint m_postProcessingProgram;
    gl::GLuint m_postProcessingVertexShader;
    gl::GLuint m_postProcessingFragmentShader;
};
