
#pragma once

#include <vector>

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <glbinding/gl/types.h>

#include "Block.h"
#include "BlockWorldImplementation.h"


class BlockWorldInstancing : public BlockWorldImplementation
{
public:
    BlockWorldInstancing();
    ~BlockWorldInstancing();

    virtual void onInitialize() override;
    virtual void onRender() override;

    virtual bool loadShader() override;

    virtual void setBlock(size_t index, const Block & block) override;

    virtual size_t size() const override;
    virtual size_t verticesCount() const override;
    virtual size_t staticByteSize() const override;
    virtual size_t byteSize() const override;
    virtual size_t vertexByteSize() const override;
    virtual size_t componentCount() const override;

    virtual void resize(size_t count) override;

    virtual gl::GLuint program() const override;
public:
    std::vector<glm::ivec4> m_positionAndType;

    gl::GLuint m_vertices;
    gl::GLuint m_attributes;
    gl::GLuint m_vao;

    gl::GLuint m_vertexShader;
    gl::GLuint m_fragmentShader;

    gl::GLuint m_program;

    void initializeVAO();
    size_t verticesPerCuboid() const;
};
