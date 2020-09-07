
#include <chrono>
#include <vector>

#include <glm/vec3.hpp>

#include <glbinding/gl/types.h>


class Implementation;
class Postprocessing;
class Screenshot;


class Rendering
{
public:
    Rendering(const std::string & name);
    virtual ~Rendering();

    void initialize();
    void deinitialize();

    void resize(int w, int h);
    void render();

    void setCameraTechnique(int i);
    void setTechnique(int i);
    void toggleRasterizerDiscard();
    void togglePostprocessing();
    void spaceMeasurement();
    void reloadShaders();
    void startFPSMeasuring();
    void startPerformanceMeasuring();
    void takeScreenshot();

    void setGridSize(int gridSize);

protected:
    std::string m_name;

    // Scene setup
    int m_cameraSetting;
    Implementation * m_current;
    Postprocessing * m_postprocessing;
    Screenshot * m_screenshot;
    std::vector<Implementation *> m_implementations;

    int m_width;
    int m_height;
    int m_gridSize;

    bool m_usePostprocessing;
    bool m_rasterizerDiscard;

    // Performance helper
    std::chrono::high_resolution_clock::time_point m_start;
    gl::GLuint m_query;

    // FPS measuring
    size_t m_fpsSamples;
    std::chrono::high_resolution_clock::time_point m_fpsMeasurementStart;

    // Performance measuring
    bool m_inMeasurement;
    size_t m_count;
    size_t m_warmupCount;
    size_t m_sum;

protected:
    void cameraPosition(glm::vec3 & eye, glm::vec3 & center, glm::vec3 & up) const;

    void prepareRendering();
    void finalizeRendering();

    void addImplementation(Implementation * implementation);

    size_t measureGPU(std::function<void()> callback, bool on) const;
    size_t measureCPU(std::function<void()> callback, bool on) const;
    void measureGPU(const std::string & name, std::function<void()> callback, bool on) const;
    void measureCPU(const std::string & name, std::function<void()> callback, bool on) const;

    size_t primitiveCount();

protected:
    // Subclass interface

    virtual void onInitialize() = 0;
    virtual void onDeinitialize() = 0;
    virtual void onCreateGeometry() = 0;
    virtual void onPrepareRendering() = 0;
    virtual void onFinalizeRendering() = 0;
};
