
#pragma once

#include <vector>

#include <glm/vec2.hpp>

#include <glbinding/gl/types.h>

#include "Arc.h"
#include "ArcImplementation.h"


class ArcVertexCloud : public ArcImplementation
{
public:
    ArcVertexCloud(bool useAlternativeShaders);
    ~ArcVertexCloud();

    virtual void onInitialize() override;
    virtual void onRender() override;

    virtual bool loadShader() override;

    virtual void setArc(size_t index, const Arc & arc) override;

    virtual size_t size() const override;
    virtual size_t verticesCount() const override;
    virtual size_t staticByteSize() const override;
    virtual size_t byteSize() const override;
    virtual size_t vertexByteSize() const override;
    virtual size_t componentCount() const override;

    virtual void resize(size_t count) override;

    virtual gl::GLuint program() const override;
public:
    bool m_alternativeShaders;

    std::vector<glm::vec2> m_center;
    std::vector<glm::vec2> m_heightRange;
    std::vector<glm::vec2> m_angleRange;
    std::vector<glm::vec2> m_radiusRange;
    std::vector<float> m_colorValue;
    std::vector<int> m_tessellationCount;

    gl::GLuint m_vertices;
    gl::GLuint m_vao;

    gl::GLuint m_vertexShader;
    gl::GLuint m_tessControlShader;
    gl::GLuint m_tessEvaluationShader;
    gl::GLuint m_geometryShader;
    gl::GLuint m_fragmentShader;

    gl::GLuint m_program;

    void initializeVAO();
    size_t verticesPerArc() const;
};
