
#include "common.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <utility>
#include <algorithm>

#include <glm/vec4.hpp>
#include <glm/exponential.hpp>
#include <glm/geometric.hpp>

#include <cpplocate/cpplocate.h>

#include <glbinding/gl32core/gl.h>  // this is a OpenGL feature include; it declares all OpenGL 3.2 Core symbols

using namespace gl;

// Read raw binary file into a char vector (probably the fastest way).


namespace
{


std::string determineDataPath()
{
    std::string path = cpplocate::locatePath("data/shaders", "share/attributedvertexclouds/shaders", reinterpret_cast<void *>(&dataPath));
    if (path.empty()) path = "./data";
    else              path = path + "data";

    return path;
}


} // namespace


const std::string & dataPath()
{
    static const auto path = determineDataPath();

    return path;
}

std::vector<char> rawFromFile(const std::string & filePath)
{
    auto stream = std::ifstream(filePath, std::ios::in | std::ios::binary | std::ios::ate);

    if (!stream)
    {
        std::cerr << "Reading from file '" << filePath << "' failed." << std::endl;
        return std::vector<char>();
    }

    stream.seekg(0, std::ios::end);

    const auto size = stream.tellg();
    auto raw = std::vector<char>(size);

    stream.seekg(0, std::ios::beg);
    stream.read(raw.data(), size);

    return raw;
}

std::vector<float> rawFromFileF(const std::string & filePath)
{
    auto stream = std::ifstream(filePath.c_str(), std::ios::in | std::ios::binary | std::ios::ate);

    if (!stream)
    {
        std::cerr << "Reading from file '" << filePath << "' failed." << std::endl;
        return std::vector<float>();
    }

    stream.seekg(0, std::ios::end);

    const auto size = stream.tellg();
    auto raw = std::vector<float>(size / sizeof(float));

    stream.seekg(0, std::ios::beg);
    stream.read(reinterpret_cast<char *>(raw.data()), (size / sizeof(float)) * sizeof(float));

    return raw;
}

std::string textFromFile(const std::string & filePath)
{
    const auto text = rawFromFile(filePath);
    return std::string(text.begin(), text.end());
}

std::string loadShaderSource(const std::string & shaderPath)
{
    return textFromFile(dataPath() + "/shaders" + shaderPath);
}

std::vector<float> loadNoise(const std::string & noisePath)
{
    return rawFromFileF(dataPath() + "/noise" + noisePath);
}

bool checkForCompilationError(GLuint shader, const std::string & identifier)
{
    auto success = static_cast<GLint>(GL_FALSE);
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

    if (success != static_cast<GLint>(GL_FALSE))
        return true;

    auto length = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);

    std::vector<char> log(length);

    glGetShaderInfoLog(shader, length, &length, log.data());

    std::cerr
        << "Compiler error in " << identifier << ":" << std::endl
        << std::string(log.data(), length) << std::endl;

    return false;
}

bool rawToFile(const char * filePath, const std::vector<char> & raw)
{
    auto stream = std::ofstream(filePath, std::ios::out | std::ios::binary);

    if (!stream)
    {
        std::cerr << "Writing to file '" << filePath << "' failed." << std::endl;
        return false;
    }

    stream.write(raw.data(), raw.size());

    return true;
}

bool checkForLinkerError(GLuint program, const std::string & identifier)
{
    auto success = static_cast<GLint>(GL_FALSE);
    glGetProgramiv(program, GL_LINK_STATUS, &success);

    if (success != static_cast<GLint>(GL_FALSE))
        return true;

    auto length = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);

    std::vector<char> log(length);

    glGetProgramInfoLog(program, length, &length, log.data());

    std::cerr
        << "Linker error in " << identifier << ":" << std::endl
        << std::string(log.data(), length) << std::endl;

    return false;

}

glm::vec3 cameraPath(const glm::vec3 & eye, float f)
{
    auto eyeRotation = glm::mat4(1.0f);
    //eyeRotation = glm::rotate(eyeRotation, glm::sin(0.8342378f * f), glm::vec3(0.0f, 1.0f, 0.0f));
    //eyeRotation = glm::rotate(eyeRotation, glm::cos(-0.5423543f * f), glm::vec3(1.0f, 0.0f, 0.0f));
    //eyeRotation = glm::rotate(eyeRotation, glm::sin(0.13234823f * f), glm::vec3(0.0f, 0.0f, 1.0f));

    eyeRotation = glm::rotate(eyeRotation, 0.1f * f, glm::vec3(0.0f, 1.0f, 0.0f));
    eyeRotation = glm::rotate(eyeRotation, glm::cos(0.14583f * f), glm::vec3(0.0f, 0.0f, 1.0f));
    eyeRotation = glm::scale(eyeRotation, glm::vec3(0.75f + 0.25f * glm::cos(0.353478534f * f)));

    return glm::vec3(eyeRotation * glm::vec4(eye, 1.0f));
}
