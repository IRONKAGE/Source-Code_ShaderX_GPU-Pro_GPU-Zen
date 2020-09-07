
#pragma once

#include <vector>

#include <glm/vec2.hpp>

#include <glbinding/gl/types.h>

#include "Implementation.h"

#include "TrajectoryNode.h"


class TrajectoryVertexCloud : public Implementation
{
public:
    TrajectoryVertexCloud();
    virtual ~TrajectoryVertexCloud();

    virtual void onInitialize() override;
    virtual void onRender() override;

    virtual bool loadShader() override;

    virtual size_t size() const override;
    virtual size_t verticesCount() const override;
    virtual size_t staticByteSize() const override;
    virtual size_t byteSize() const override;
    virtual size_t vertexByteSize() const override;
    virtual size_t componentCount() const override;

    virtual void resize(size_t count) override;

    virtual gl::GLuint program() const override;

    void setTrajectoryNode(size_t index, const TrajectoryNode & node);

public:
    std::vector<glm::vec3> m_position;
    std::vector<int> m_trajectoryID;
    std::vector<int> m_type;
    std::vector<glm::vec3> m_incoming;
    std::vector<glm::vec3> m_outgoing;
    std::vector<float> m_colorValue;
    std::vector<float> m_sizeValue;

    gl::GLuint m_vertices;
    gl::GLuint m_vao;

    gl::GLuint m_vertexShader;
    gl::GLuint m_tessControlShader;
    gl::GLuint m_tessEvaluationShader;
    gl::GLuint m_geometryShader;
    gl::GLuint m_fragmentShader;

    gl::GLuint m_program;

    void initializeVAO();
    size_t verticesPerNode() const;
};
