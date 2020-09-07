
#pragma once

#include <vector>
#include <mutex>

#include <glm/vec2.hpp>

#include <glbinding/gl/types.h>

#include "Polygon.h"
#include "PolygonImplementation.h"


class PolygonVertexCloud : public PolygonImplementation
{
public:
    PolygonVertexCloud();
    ~PolygonVertexCloud();

    virtual void onInitialize() override;
    virtual void onRender() override;

    virtual bool loadShader() override;

    virtual void setPolygon(size_t index, const Polygon & polygon) override;

    virtual size_t size() const override;
    virtual size_t verticesCount() const override;
    virtual size_t staticByteSize() const override;
    virtual size_t byteSize() const override;
    virtual size_t vertexByteSize() const override;
    virtual size_t componentCount() const override;

    virtual void resize(size_t count) override;

    virtual gl::GLuint program() const override;
public:
    std::vector<glm::vec2> m_center;
    std::vector<glm::vec2> m_heightRange;
    std::vector<float> m_colorValue;

    std::vector<glm::vec2> m_positions;
    std::vector<int> m_polygonIndices;

    gl::GLuint m_vertices;
    gl::GLuint m_centerHeightRangeBuffer;
    gl::GLuint m_colorValueBuffer;
    gl::GLuint m_centerHeightRangeTexture;
    gl::GLuint m_colorValueTexture;

    gl::GLuint m_vao;

    gl::GLuint m_vertexShader;
    gl::GLuint m_geometryShader;
    gl::GLuint m_fragmentShader;

    gl::GLuint m_program;

    void initializeVAO();

protected:
    std::mutex m_mutex;
};
