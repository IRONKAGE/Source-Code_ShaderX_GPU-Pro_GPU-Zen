
#include "Rendering.h"

#include <iostream>
#include <algorithm>
#include <numeric>

#include <glm/gtc/type_ptr.hpp>

#include <glbinding/gl/gl.h>

#include "common.h"

#include "Implementation.h"
#include "Postprocessing.h"
#include "Screenshot.h"


using namespace gl;


namespace
{

static const auto warmupCount = size_t(1000);
static const auto measureCount = size_t(1000);
static const auto fpsSampleCount = size_t(100);

static const auto screenshotWidth = size_t(3840 * 2);
static const auto screenshotHeight = size_t(2160 * 2);

//static const float clearColor[] = { 0.8f, 0.8f, 0.8f, 1.0f };
//static const float clearColor[] = { 1.0f, 1.0f, 1.0f, 1.0f };
//static const float clearColor[] = { 0.12f, 0.14f, 0.18f, 1.0f };
static const float clearColor[] = { 0.81f, 0.81f, 0.81f, 1.0f };


} // namespace


Rendering::Rendering(const std::string & name)
: m_name(name)
, m_current(nullptr)
, m_postprocessing(nullptr)
, m_width(0)
, m_height(0)
, m_gridSize(32)
, m_usePostprocessing(false)
, m_rasterizerDiscard(false)
, m_query(0)
, m_fpsSamples(fpsSampleCount+1)
, m_inMeasurement(false)
, m_count(0)
, m_warmupCount(0)
, m_sum(0)
{
}

Rendering::~Rendering()
{
}

void Rendering::addImplementation(Implementation *implementation)
{
    m_implementations.push_back(implementation);
}

void Rendering::initialize()
{
    glClearColor(clearColor[0], clearColor[1], clearColor[2], clearColor[3]);
    glClearDepthf(1.0f);
    glEnable(GL_DEPTH_TEST);

    glGenQueries(1, &m_query);

    m_postprocessing = new Postprocessing;
    m_screenshot = new Screenshot;

    onInitialize();
    onCreateGeometry();

    m_start = std::chrono::high_resolution_clock::now();

    setTechnique(0);
}

void Rendering::deinitialize()
{
    onDeinitialize();

    // Flag all aquired resources for deletion (hint: driver decides when to actually delete them; see: shared contexts)
    glDeleteQueries(1, &m_query);

    delete m_postprocessing;

    for (auto implementation : m_implementations)
    {
        delete implementation;
    }
}

void Rendering::reloadShaders()
{
    if (m_postprocessing->initialized())
    {
        m_postprocessing->loadShader();
    }

    for (auto implementation : m_implementations)
    {
        if (implementation->initialized())
        {
            implementation->loadShader();
        }
    }
}

void Rendering::cameraPosition(glm::vec3 & eye, glm::vec3 & center, glm::vec3 & up) const
{
    static const auto eye0 = glm::vec3(1.1f, 1.1f, 1.1f);
    static const auto eye1 = glm::vec3(1.2f, 0.0f, 1.2f);
    static const auto eye2 = glm::vec3(1.4f, 0.0f, 0.0f);
    static const auto eye3 = glm::vec3(0.9f, 0.6f, 0.6f);

    static const auto center0 = glm::vec3(0.0f, 0.0f, 0.0f);
    static const auto center1 = glm::vec3(0.1f, 0.2f, 0.0f);

    static const auto up0 = glm::vec3(0.0f, 1.0f, 0.0f);

    const auto f = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - m_start).count()) / 1000.0f;

    if (m_inMeasurement)
    {
        eye = eye0;
        center = center0;
        up = up0;

        return;
    }

    switch (m_cameraSetting)
    {
    case 0:
        eye = cameraPath(eye0, f);
        center = center0;
        up = up0;
        break;
    case 1:
        eye = eye0;
        center = center0;
        up = up0;
        break;
    case 2:
        eye = eye1;
        center = center0;
        up = up0;
        break;
    case 3:
        eye = eye2;
        center = center1;
        up = up0;
        break;
    case 4:
        eye = eye3;
        center = center1;
        up = up0;
        break;
    default:
        eye = cameraPath(eye0, f);
        center = center0;
        up = up0;
        break;
    }
}

void Rendering::prepareRendering()
{
    auto eye = glm::vec3(0.0f, 0.0f, 0.0f);
    auto center = glm::vec3(0.0f, 0.0f, 0.0f);
    auto up = glm::vec3(0.0f, 0.0f, 0.0f);

    cameraPosition(eye, center, up);

    const auto view = glm::lookAt(eye, center, up);
    const auto viewProjection = glm::perspectiveFov(glm::radians(45.0f), float(m_width), float(m_height), 0.05f, 2.5f) * view;

    GLuint program = m_current->program();
    const auto viewProjectionLocation = glGetUniformLocation(program, "viewProjection");
    const auto gradientSamplerLocation = glGetUniformLocation(program, "gradient");
    glUseProgram(program);
    glUniformMatrix4fv(viewProjectionLocation, 1, GL_FALSE, glm::value_ptr(viewProjection));
    glUniform1i(gradientSamplerLocation, 0);

    glUseProgram(0);

    onPrepareRendering();
}

void Rendering::finalizeRendering()
{
    onFinalizeRendering();
}

void Rendering::resize(int w, int h)
{
    m_width = w;
    m_height = h;

    if (m_postprocessing && m_postprocessing->initialized())
    {
        m_postprocessing->resize(m_width, m_height);
    }
}

void Rendering::setCameraTechnique(int i)
{
    if (i < 0 || i >= 4)
    {
        return;
    }

    m_cameraSetting = i;
}

void Rendering::setTechnique(int i)
{
    if (i < 0 || static_cast<std::size_t>(i) >= m_implementations.size())
    {
        return;
    }

    m_current = m_implementations.at(static_cast<std::size_t>(i));

    std::cout << "Switch to " << m_current->name() << " implementation" << std::endl;
}

void Rendering::render()
{
    if (m_inMeasurement)
    {
        m_current->initialize();

        glViewport(0, 0, m_width, m_height);

        glEnable(GL_RASTERIZER_DISCARD);

        prepareRendering();

        m_sum += measureGPU([this]() {
            m_current->render();
        }, m_warmupCount == 0);

        if (m_warmupCount > 0)
        {
            --m_warmupCount;
        }
        else
        {
            ++m_count;

            if (m_count == measureCount)
            {
                m_inMeasurement = false;

                std::cout << "Measured " << (m_sum / m_count / 1000) << "µs" << " for geometry processing" << std::endl;
            }
        }

        finalizeRendering();

        glDisable(GL_RASTERIZER_DISCARD);

        return;
    }

    if (m_fpsSamples == fpsSampleCount)
    {
        const auto end = std::chrono::high_resolution_clock::now();

        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - m_fpsMeasurementStart).count() / 1000.0f / fpsSampleCount;

        std::cout << "Measured " << (1.0f / elapsed) << "FPS (" << "(~ " << (elapsed * 1000.0f) << "ms per frame)" << std::endl;

        m_fpsSamples = fpsSampleCount + 1;
    }

    if (m_fpsSamples < fpsSampleCount)
    {
        ++m_fpsSamples;
    }

    m_current->initialize();

    glViewport(0, 0, m_width, m_height);

    if (m_usePostprocessing && !m_rasterizerDiscard)
    {
        if (!m_postprocessing->initialized())
        {
            m_postprocessing->initialize();
        }

        m_postprocessing->resize(m_width, m_height);

        glBindFramebuffer(GL_FRAMEBUFFER, m_postprocessing->fbo());

        glClearBufferfv(GL_COLOR, 0, clearColor);
        glClearBufferfi(GL_DEPTH_STENCIL, 0, 1.0f, 0);
    }
    else
    {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }

    prepareRendering();

    if (m_rasterizerDiscard)
    {
        glEnable(GL_RASTERIZER_DISCARD);
    }

    m_current->render();

    if (m_rasterizerDiscard)
    {
        glDisable(GL_RASTERIZER_DISCARD);
    }

    finalizeRendering();

    if (m_usePostprocessing && !m_rasterizerDiscard)
    {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        m_postprocessing->render();
    }
}

void Rendering::takeScreenshot()
{
    if (m_inMeasurement || m_fpsSamples < fpsSampleCount)
    {
        std::cout << "No screenshot during measurements allowed" << std::endl;

        return;
    }

    if (!m_screenshot->initialized())
    {
        m_screenshot->initialize();
        m_screenshot->resize(screenshotWidth, screenshotHeight);
    }

    m_current->initialize();

    glViewport(0, 0, screenshotWidth, screenshotHeight);

    if (!m_postprocessing->initialized())
    {
        m_postprocessing->initialize();
    }

    m_postprocessing->resize(screenshotWidth, screenshotHeight);

    glBindFramebuffer(GL_FRAMEBUFFER, m_postprocessing->fbo());

    glClearBufferfv(GL_COLOR, 0, clearColor);
    glClearBufferfi(GL_DEPTH_STENCIL, 0, 1.0f, 0);

    prepareRendering();

    m_current->render();

    finalizeRendering();

    glBindFramebuffer(GL_FRAMEBUFFER, m_screenshot->fbo());

    m_postprocessing->render();

    m_screenshot->saveScreenshot(m_name + "-" + std::to_string(m_gridSize) + ".ppm");
}

void Rendering::spaceMeasurement()
{
    const auto reference = std::accumulate(m_implementations.begin(), m_implementations.end(),
            std::accumulate(m_implementations.begin(), m_implementations.end(), 0, [](size_t currentSize, const Implementation * technique) {
                return std::max(currentSize, technique->fullByteSize());
            }), [](size_t currentSize, const Implementation * technique) {
        return std::min(currentSize, technique->fullByteSize());
    });

    const auto printSpaceMeasurement = [&reference](const std::string & techniqueName, size_t byteSize)
    {
        std::cout << techniqueName << std::endl << (byteSize / 1024) << "kB (" << (static_cast<float>(byteSize) / reference) << "x)" << std::endl;
    };

    std::cout << "Count: " << primitiveCount() << std::endl;
    std::cout << std::endl;

    for (const auto implementation : m_implementations)
    {
        printSpaceMeasurement(implementation->name(), implementation->fullByteSize());
    }
}

size_t Rendering::measureGPU(std::function<void()> callback, bool on) const
{
    if (!on)
    {
        callback();

        return 0;
    }

    glBeginQuery(gl::GL_TIME_ELAPSED, m_query);

    callback();

    glEndQuery(gl::GL_TIME_ELAPSED);

    int available = 0;
    while (!available)
    {
        glGetQueryObjectiv(m_query, gl::GL_QUERY_RESULT_AVAILABLE, &available);
    }

    int value;
    glGetQueryObjectiv(m_query, gl::GL_QUERY_RESULT, &value);

    return static_cast<std::size_t>(value);
}

size_t Rendering::measureCPU(std::function<void()> callback, bool on) const
{
    if (!on)
    {
        callback();

        return 0;
    }

    const auto start = std::chrono::high_resolution_clock::now();

    callback();

    const auto end = std::chrono::high_resolution_clock::now();

    return static_cast<std::size_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

void Rendering::measureCPU(const std::string & name, std::function<void()> callback, bool on) const
{
    if (!on)
    {
        return callback();
    }

    std::cout << name << ": " << measureCPU(callback, on) << "ns" << std::endl;
}

void Rendering::measureGPU(const std::string & name, std::function<void()> callback, bool on) const
{
    if (!on)
    {
        return callback();
    }

    std::cout << name << ": " << measureGPU(callback, on) << "ns" << std::endl;
}

void Rendering::toggleRasterizerDiscard()
{
    m_rasterizerDiscard = !m_rasterizerDiscard;
}

void Rendering::startFPSMeasuring()
{
    m_fpsSamples = 0;
    m_fpsMeasurementStart = std::chrono::high_resolution_clock::now();
}

void Rendering::startPerformanceMeasuring()
{
    m_inMeasurement = true;
    m_count = 0;
    m_warmupCount = warmupCount;
    m_sum = 0;
}

void Rendering::togglePostprocessing()
{
    m_usePostprocessing = !m_usePostprocessing;
}

void Rendering::setGridSize(int gridSize)
{
    m_gridSize = gridSize;
}

size_t Rendering::primitiveCount()
{
    return static_cast<std::size_t>(m_gridSize * m_gridSize * m_gridSize);
}
