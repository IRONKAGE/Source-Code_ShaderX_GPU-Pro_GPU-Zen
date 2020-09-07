
#pragma once

#include <string>

#include <glbinding/gl/types.h>

class Implementation
{
public:
    Implementation(const std::string & name);
    virtual ~Implementation();

    const std::string & name() const;

    bool initialized() const;

    void initialize();
    void render();

    size_t fullByteSize() const;

    virtual void onInitialize() = 0;
    virtual void onRender() = 0;

    virtual bool loadShader() = 0;

    virtual size_t size() const = 0;
    virtual size_t verticesCount() const = 0;
    virtual size_t staticByteSize() const = 0;
    virtual size_t byteSize() const = 0;
    virtual size_t vertexByteSize() const = 0;
    virtual size_t componentCount() const = 0;

    virtual void resize(size_t count) = 0;

    virtual gl::GLuint program() const = 0;

protected:
    std::string m_name;
    bool m_initialized;
};
