/* $Id: shader.cpp 247 2009-09-08 08:57:56Z chomzee $ */

#include <Cg/cg.h>

#ifdef RNDR_D3D
	#include <d3dx9.h>
	#include <Cg/cgD3D9.h>
#else
	#include <Cg/cgGL.h>
#endif

#include <stdio.h>
#include <string.h>

#include "shader.h"
#include "renderer.h"
#include "../math/blossom_engine_math.h"
#include "../common/blossom_engine_common.h"

// ----------------------------------------------------------------------------

namespace Blossom
{
	int CShader::referenceCounter = 0;

	CGparameter CShader::cgParameter;



	void CShader::init(std::string fileName, ShaderType shaderType, int argsNum, const char **args)
	{
		if (exists)
			return;

		#ifdef RNDR_D3D
		{
			if (shaderType == stVertexShader)
				cgProgram = cgCreateProgramFromFile(CRenderer::cgContext, CG_SOURCE, fileName.c_str(), CG_PROFILE_VS_3_0, "main", args);
			else
				cgProgram = cgCreateProgramFromFile(CRenderer::cgContext, CG_SOURCE, fileName.c_str(), CG_PROFILE_PS_3_0, "main", args);
		}
		#else
		{
			if (CRenderer::isNVidiaGPU)
			{
				if (shaderType == stVertexShader)
					cgProgram = cgCreateProgramFromFile(CRenderer::cgContext, CG_SOURCE, fileName.c_str(), CG_PROFILE_VP40, "main", args);
				else
					cgProgram = cgCreateProgramFromFile(CRenderer::cgContext, CG_SOURCE, fileName.c_str(), CG_PROFILE_FP40, "main", args);
			}
			else
			{
				if (shaderType == stVertexShader)
					cgProgram = cgCreateProgramFromFile(CRenderer::cgContext, CG_SOURCE, fileName.c_str(), CG_PROFILE_ARBVP1, "main", args);
				else
					cgProgram = cgCreateProgramFromFile(CRenderer::cgContext, CG_SOURCE, fileName.c_str(), CG_PROFILE_ARBFP1, "main", args);
			}
		}
		#endif

		this->shaderType = shaderType;

		const char *asmSource;
		const char *errors;

		errors = cgGetLastListing(CRenderer::cgContext);

		if (errors)
		{
			CLogger::addText("\tERROR: shader errors while compiling, %s\n", fileName.c_str());

			char shaderErrorsFileName[100];
			sprintf(shaderErrorsFileName, "%s_", fileName.c_str());
			for (int i = 0; i < argsNum; i++)
			{
				char argName[100];
				sprintf(argName, args[i]);
				strcat(shaderErrorsFileName, argName);
			}
			strcat(shaderErrorsFileName, "_ERRORS.txt");
			{
				FILE *file = fopen(shaderErrorsFileName, "wt");
					fprintf(file, errors);
				fclose(file);
			}

			exit(1);
		}
		else
		{
			if (cgProgram == 0)
			{
				CLogger::addText("\tERROR: couldn't load shader from file, %s\n", fileName.c_str());
				exit(1);
			}

			asmSource = cgGetProgramString(cgProgram, CG_COMPILED_PROGRAM);

			char shaderAsmSourceFileName[1024];
			sprintf(shaderAsmSourceFileName, "%s_", fileName.c_str());
			for (int i = 0; i < argsNum; i++)
			{
				char argName[100];
				sprintf(argName, args[i]);
				strcat(shaderAsmSourceFileName, argName);
			}
			strcat(shaderAsmSourceFileName, "_asm_source.txt");
			{
				FILE *file = fopen(shaderAsmSourceFileName, "wt");
					fprintf(file, asmSource);
				fclose(file);
			}
		}

		#ifdef RNDR_D3D
		{
			cgD3D9LoadProgram(cgProgram, false, 0);
		}
		#else
		{
			cgGLLoadProgram(cgProgram);
		}
		#endif

		exists = true;
		referenceCounter++;
		CLogger::addText("\tOK: shader compiled and loaded, %s (reference counter = %d)\n", fileName.c_str(), referenceCounter);
	}



	void CShader::free()
	{
		if (!exists)
			return;

		cgDestroyProgram(cgProgram);

		exists = false;
		referenceCounter--;
		CLogger::addText("\tOK: shader freed (reference counter = %d)\n", referenceCounter);
	}



	ShaderType CShader::getShaderType() const
	{
		return shaderType;
	}



	// ----------------------------------------------------------------------------



	void CShader::setVertexShaderFloatConstant(const char *name, float x)
	{
		cgParameter = cgGetNamedParameter(CRenderer::currentVertexShader->cgProgram, name);

		#ifdef RNDR_D3D
		{
			float floats[] = { x };
			cgD3D9SetUniform(cgParameter, floats);
		}
		#else
		{
			cgGLSetParameter1f(cgParameter, x);
		}
		#endif
	}



	void CShader::setVertexShaderVectorConstant(const char *name, const CVector2 &v, float z, float w)
	{
		cgParameter = cgGetNamedParameter(CRenderer::currentVertexShader->cgProgram, name);

		#ifdef RNDR_D3D
		{
			float floats[] = { v.x, v.y, z, w };
			cgD3D9SetUniform(cgParameter, floats);
		}
		#else
		{
			cgGLSetParameter4f(cgParameter, v.x, v.y, z, w);
		}
		#endif
	}



	void CShader::setVertexShaderVectorConstant(const char *name, const CVector3 &v, float w)
	{
		cgParameter = cgGetNamedParameter(CRenderer::currentVertexShader->cgProgram, name);

		#ifdef RNDR_D3D
		{
			float floats[] = { v.x, v.y, v.z, w };
			cgD3D9SetUniform(cgParameter, floats);
		}
		#else
		{
			cgGLSetParameter4f(cgParameter, v.x, v.y, v.z, w);
		}
		#endif
	}



	void CShader::setVertexShaderVectorConstant(const char *name, const CVector4 &v)
	{
		cgParameter = cgGetNamedParameter(CRenderer::currentVertexShader->cgProgram, name);

		#ifdef RNDR_D3D
		{
			float floats[] = { v.x, v.y, v.z, v.w };
			cgD3D9SetUniform(cgParameter, floats);
		}
		#else
		{
			cgGLSetParameter4f(cgParameter, v.x, v.y, v.z, v.w);
		}
		#endif
	}



	void CShader::setVertexShaderVectorConstant(const char *name, float x, float y, float z, float w)
	{
		cgParameter = cgGetNamedParameter(CRenderer::currentVertexShader->cgProgram, name);

		#ifdef RNDR_D3D
		{
			float floats[] = { x, y, z, w };
			cgD3D9SetUniform(cgParameter, floats);
		}
		#else
		{
			cgGLSetParameter4f(cgParameter, x, y, z, w);
		}
		#endif
	}



	void CShader::setVertexShaderVectorArrayConstants(const char *name, CVector2 *v, float *z, float *w, int count)
	{
		float *values = new float[4*count];

		for (int i = 0; i < count; i++)
		{
			values[4*i + 0] = v[i].x;
			values[4*i + 1] = v[i].y;
			values[4*i + 2] = z[i];
			values[4*i + 3] = w[i];
		}

		cgParameter = cgGetNamedParameter(CRenderer::currentVertexShader->cgProgram, name);

		#ifdef RNDR_D3D
		{
			cgD3D9SetUniformArray(cgParameter, 0, count, values);
		}
		#else
		{
			cgGLSetParameterArray4f(cgParameter, 0, count, values);
		}
		#endif

		delete[] values;
	}



	void CShader::setVertexShaderVectorArrayConstants(const char *name, CVector3 *v, float *w, int count)
	{
		float *values = new float[4*count];

		for (int i = 0; i < count; i++)
		{
			values[4*i + 0] = v[i].x;
			values[4*i + 1] = v[i].y;
			values[4*i + 2] = v[i].z;
			values[4*i + 3] = w[i];
		}

		cgParameter = cgGetNamedParameter(CRenderer::currentVertexShader->cgProgram, name);

		#ifdef RNDR_D3D
		{
			cgD3D9SetUniformArray(cgParameter, 0, count, values);
		}
		#else
		{
			cgGLSetParameterArray4f(cgParameter, 0, count, values);
		}
		#endif

		delete[] values;
	}



	void CShader::setVertexShaderVectorArrayConstants(const char *name, CVector4 *v, int count)
	{
		float *values = new float[4*count];

		for (int i = 0; i < count; i++)
		{
			values[4*i + 0] = v[i].x;
			values[4*i + 1] = v[i].y;
			values[4*i + 2] = v[i].z;
			values[4*i + 3] = v[i].w;
		}

		cgParameter = cgGetNamedParameter(CRenderer::currentVertexShader->cgProgram, name);

		#ifdef RNDR_D3D
		{
			cgD3D9SetUniformArray(cgParameter, 0, count, values);
		}
		#else
		{
			cgGLSetParameterArray4f(cgParameter, 0, count, values);
		}
		#endif

		delete[] values;
	}



	void CShader::setVertexShaderVectorArrayConstants(const char *name, float *x, float *y, float *z, float *w, int count)
	{
		float *values = new float[4*count];

		for (int i = 0; i < count; i++)
		{
			values[4*i + 0] = x[i];
			values[4*i + 1] = y[i];
			values[4*i + 2] = z[i];
			values[4*i + 3] = w[i];
		}

		cgParameter = cgGetNamedParameter(CRenderer::currentVertexShader->cgProgram, name);

		#ifdef RNDR_D3D
		{
			cgD3D9SetUniformArray(cgParameter, 0, count, values);
		}
		#else
		{
			cgGLSetParameterArray4f(cgParameter, 0, count, values);
		}
		#endif

		delete[] values;
	}



	void CShader::setVertexShaderMatrixConstant(const char *name, const CMatrix &m)
	{
		cgParameter = cgGetNamedParameter(CRenderer::currentVertexShader->cgProgram, name);

		#ifdef RNDR_D3D
		{
			D3DXMATRIX matrix;
			memcpy((void*)&matrix, (void*)&m, 64);

			cgD3D9SetUniformMatrix(cgParameter, &matrix);
		}
		#else
		{
			float matrix[16];
			memcpy((void*)matrix, (void*)&m, 64);

			cgGLSetMatrixParameterfr(cgParameter, matrix);
		}
		#endif
	}



	void CShader::setVertexShaderMatrixArrayConstants(const char *name, CMatrix *m, int count)
	{
		cgParameter = cgGetNamedParameter(CRenderer::currentVertexShader->cgProgram, name);

		#ifdef RNDR_D3D
		{
			D3DXMATRIX *matrices = new D3DXMATRIX[count];
			memcpy((void*)matrices, (void*)m, count*64);

			cgD3D9SetUniformMatrixArray(cgParameter, 0, count, matrices);

			delete[] matrices;
		}
		#else
		{
			float *matrices = new float[count*16];
			memcpy((void*)matrices, (void*)m, count*64);

			cgGLSetMatrixParameterArrayfr(cgParameter, 0, count, matrices);

			delete matrices;
		}
		#endif
	}



	// ----------------------------------------------------------------------------



	void CShader::setPixelShaderFloatConstant(const char *name, float x)
	{
		cgParameter = cgGetNamedParameter(CRenderer::currentPixelShader->cgProgram, name);

		#ifdef RNDR_D3D
		{
			float floats[] = { x };
			cgD3D9SetUniform(cgParameter, floats);
		}
		#else
		{
			cgGLSetParameter1f(cgParameter, x);
		}
		#endif
	}



	void CShader::setPixelShaderVectorConstant(const char *name, const CVector2 &v, float z, float w)
	{
		cgParameter = cgGetNamedParameter(CRenderer::currentPixelShader->cgProgram, name);

		#ifdef RNDR_D3D
		{
			float floats[] = { v.x, v.y, z, w };
			cgD3D9SetUniform(cgParameter, floats);
		}
		#else
		{
			cgGLSetParameter4f(cgParameter, v.x, v.y, z, w);
		}
		#endif
	}



	void CShader::setPixelShaderVectorConstant(const char *name, const CVector3 &v, float w)
	{
		cgParameter = cgGetNamedParameter(CRenderer::currentPixelShader->cgProgram, name);

		#ifdef RNDR_D3D
		{
			float floats[] = { v.x, v.y, v.z, w };
			cgD3D9SetUniform(cgParameter, floats);
		}
		#else
		{
			cgGLSetParameter4f(cgParameter, v.x, v.y, v.z, w);
		}
		#endif
	}



	void CShader::setPixelShaderVectorConstant(const char *name, const CVector4 &v)
	{
		cgParameter = cgGetNamedParameter(CRenderer::currentPixelShader->cgProgram, name);

		#ifdef RNDR_D3D
		{
			float floats[] = { v.x, v.y, v.z, v.w };
			cgD3D9SetUniform(cgParameter, floats);
		}
		#else
		{
			cgGLSetParameter4f(cgParameter, v.x, v.y, v.z, v.w);
		}
		#endif
	}



	void CShader::setPixelShaderVectorConstant(const char *name, float x, float y, float z, float w)
	{
		cgParameter = cgGetNamedParameter(CRenderer::currentPixelShader->cgProgram, name);

		#ifdef RNDR_D3D
		{
			float floats[] = { x, y, z, w };
			cgD3D9SetUniform(cgParameter, floats);
		}
		#else
		{
			cgGLSetParameter4f(cgParameter, x, y, z, w);
		}
		#endif
	}



	void CShader::setPixelShaderVectorArrayConstants(const char *name, CVector2 *v, float *z, float *w, int count)
	{
		float *values = new float[4*count];

		for (int i = 0; i < count; i++)
		{
			values[4*i + 0] = v[i].x;
			values[4*i + 1] = v[i].y;
			values[4*i + 2] = z[i];
			values[4*i + 3] = w[i];
		}

		cgParameter = cgGetNamedParameter(CRenderer::currentPixelShader->cgProgram, name);

		#ifdef RNDR_D3D
		{
			cgD3D9SetUniformArray(cgParameter, 0, count, values);
		}
		#else
		{
			cgGLSetParameterArray4f(cgParameter, 0, count, values);
		}
		#endif

		delete[] values;
	}



	void CShader::setPixelShaderVectorArrayConstants(const char *name, CVector3 *v, float *w, int count)
	{
		float *values = new float[4*count];

		for (int i = 0; i < count; i++)
		{
			values[4*i + 0] = v[i].x;
			values[4*i + 1] = v[i].y;
			values[4*i + 2] = v[i].z;
			values[4*i + 3] = w[i];
		}

		cgParameter = cgGetNamedParameter(CRenderer::currentPixelShader->cgProgram, name);

		#ifdef RNDR_D3D
		{
			cgD3D9SetUniformArray(cgParameter, 0, count, values);
		}
		#else
		{
			cgGLSetParameterArray4f(cgParameter, 0, count, values);
		}
		#endif

		delete[] values;
	}



	void CShader::setPixelShaderVectorArrayConstants(const char *name, CVector4 *v, int count)
	{
		float *values = new float[4*count];

		for (int i = 0; i < count; i++)
		{
			values[4*i + 0] = v[i].x;
			values[4*i + 1] = v[i].y;
			values[4*i + 2] = v[i].z;
			values[4*i + 3] = v[i].w;
		}

		cgParameter = cgGetNamedParameter(CRenderer::currentPixelShader->cgProgram, name);

		#ifdef RNDR_D3D
		{
			cgD3D9SetUniformArray(cgParameter, 0, count, values);
		}
		#else
		{
			cgGLSetParameterArray4f(cgParameter, 0, count, values);
		}
		#endif

		delete[] values;
	}



	void CShader::setPixelShaderVectorArrayConstants(const char *name, float *x, float *y, float *z, float *w, int count)
	{
		float *values = new float[4*count];

		for (int i = 0; i < count; i++)
		{
			values[4*i + 0] = x[i];
			values[4*i + 1] = y[i];
			values[4*i + 2] = z[i];
			values[4*i + 3] = w[i];
		}

		cgParameter = cgGetNamedParameter(CRenderer::currentPixelShader->cgProgram, name);

		#ifdef RNDR_D3D
		{
			cgD3D9SetUniformArray(cgParameter, 0, count, values);
		}
		#else
		{
			cgGLSetParameterArray4f(cgParameter, 0, count, values);
		}
		#endif

		delete[] values;
	}



	void CShader::setPixelShaderMatrixConstant(const char *name, const CMatrix &m)
	{
		cgParameter = cgGetNamedParameter(CRenderer::currentPixelShader->cgProgram, name);

		#ifdef RNDR_D3D
		{
			D3DXMATRIX matrix;
			memcpy((void*)&matrix, (void*)&m, 64);

			cgD3D9SetUniformMatrix(cgParameter, &matrix);
		}
		#else
		{
			float matrix[16];
			memcpy((void*)matrix, (void*)&m, 64);

			cgGLSetMatrixParameterfr(cgParameter, matrix);
		}
		#endif
	}



	void CShader::setPixelShaderMatrixArrayConstants(const char *name, CMatrix *m, int count)
	{
		cgParameter = cgGetNamedParameter(CRenderer::currentPixelShader->cgProgram, name);

		#ifdef RNDR_D3D
		{
			D3DXMATRIX *matrices = new D3DXMATRIX[count];
			memcpy((void*)matrices, (void*)m, count*64);

			cgD3D9SetUniformMatrixArray(cgParameter, 0, count, matrices);

			delete[] matrices;
		}
		#else
		{
			float *matrices = new float[count*16];
			memcpy((void*)matrices, (void*)m, count*64);

			cgGLSetMatrixParameterArrayfr(cgParameter, 0, count, matrices);

			delete matrices;
		}
		#endif
	}
}
