/****************************************************************************/
/* Copyright (c) 2011, Ola Olsson
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
/****************************************************************************/
#include "ComboShader.h"
#include "Assert.h"


namespace chag
{


ComboShader::ComboShader(const char *vertexShaderFileName, const char *fragmentShaderFileName, chag::SimpleShader::Context &context, const std::vector<std::string> &shadingModels)
{
	std::vector<std::string> sms = shadingModels;
	if (sms.empty())
	{
		// todo: define "SHADING_MODEL_DEFAULT" somewhere... OBJModel?
		sms.push_back("SHADING_MODEL_DEFAULT");
	}
	for(size_t i = 0; i < sms.size(); ++i)
	{
		const std::string &shadingModel = sms[i];
		context.setPreprocDef("SHADING_MODEL", shadingModel);

		context.setPreprocDef("ENABLE_ALPHA_TEST", 0);
		m_opaqueShaders[shadingModel] = new chag::SimpleShader(vertexShaderFileName, fragmentShaderFileName, context);

		context.setPreprocDef("ENABLE_ALPHA_TEST", 1);
		m_alphaTestedShaders[shadingModel] = new chag::SimpleShader(vertexShaderFileName, fragmentShaderFileName, context);
	}
	m_currentShader = 0;
}



ComboShader::~ComboShader()
{
	for (ShaderMap::iterator it = m_opaqueShaders.begin(); it != m_opaqueShaders.end(); ++it)
	{
		delete (*it).second;
	}
	for (ShaderMap::iterator it = m_alphaTestedShaders.begin(); it != m_alphaTestedShaders.end(); ++it)
	{
		delete (*it).second;
	}
}



bool ComboShader::link()
{
	bool ok = true;
	for (ShaderMap::iterator it = m_opaqueShaders.begin(); it != m_opaqueShaders.end(); ++it)
	{
		ok = (*it).second->link() && ok;
	}
	for (ShaderMap::iterator it = m_alphaTestedShaders.begin(); it != m_alphaTestedShaders.end(); ++it)
	{
		ok = (*it).second->link() && ok;
	}
	return ok;
}



void ComboShader::bindAttribLocation(GLint index, const GLchar* name)
{
	for (ShaderMap::iterator it = m_opaqueShaders.begin(); it != m_opaqueShaders.end(); ++it)
	{
		(*it).second->bindAttribLocation(index, name);
	}
	for (ShaderMap::iterator it = m_alphaTestedShaders.begin(); it != m_alphaTestedShaders.end(); ++it)
	{
		(*it).second->bindAttribLocation(index, name);
	}
}



void ComboShader::bindFragDataLocation(GLuint location, const char *name)
{
	for (ShaderMap::iterator it = m_opaqueShaders.begin(); it != m_opaqueShaders.end(); ++it)
	{
		(*it).second->bindFragDataLocation(location, name);
	}
	for (ShaderMap::iterator it = m_alphaTestedShaders.begin(); it != m_alphaTestedShaders.end(); ++it)
	{
		(*it).second->bindFragDataLocation(location, name);
	}
}



bool ComboShader::setUniformBufferSlot(const char *blockName, GLuint slotIndex)
{
	bool ok = true;
	for (ShaderMap::iterator it = m_opaqueShaders.begin(); it != m_opaqueShaders.end(); ++it)
	{
		ok = (*it).second->setUniformBufferSlot(blockName, slotIndex) && ok;
	}
	for (ShaderMap::iterator it = m_alphaTestedShaders.begin(); it != m_alphaTestedShaders.end(); ++it)
	{
		ok = (*it).second->setUniformBufferSlot(blockName, slotIndex) && ok;
	}
	return ok;
}



bool ComboShader::setUniform(const char *varName, int v)
{
	m_currentShader->end();

	for (ShaderMap::iterator it = m_opaqueShaders.begin(); it != m_opaqueShaders.end(); ++it)
	{
		(*it).second->begin();
		(*it).second->setUniform(varName, v);
		(*it).second->end();
	}
	for (ShaderMap::iterator it = m_alphaTestedShaders.begin(); it != m_alphaTestedShaders.end(); ++it)
	{
		(*it).second->begin();
		(*it).second->setUniform(varName, v);
		(*it).second->end();
	}
	m_currentShader->begin();
	return true;
}



void ComboShader::begin(bool useAlphaTest, const std::string &shadingModel)
{
	ASSERT(!m_currentShader);
	m_currentShader = useAlphaTest ? m_alphaTestedShaders[shadingModel] : m_opaqueShaders[shadingModel];
	ASSERT(m_currentShader);
	m_currentShader->begin();
}



void ComboShader::end()
{
	ASSERT(m_currentShader);
	m_currentShader->end();
	m_currentShader = 0;
}



}; // namespace chag


