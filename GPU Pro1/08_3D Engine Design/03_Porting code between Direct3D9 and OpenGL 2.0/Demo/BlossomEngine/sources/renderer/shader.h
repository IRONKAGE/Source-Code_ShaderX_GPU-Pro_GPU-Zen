/* $Id: shader.h 148 2009-08-24 17:28:38Z maxest $ */

#ifndef _BLOSSOM_ENGINE_SHADER_
#define _BLOSSOM_ENGINE_SHADER_

#include <string>

#include <Cg/cg.h>

// ----------------------------------------------------------------------------

namespace Blossom
{
	class CVector2;
	class CVector3;
	class CVector4;
	class CMatrix;
	class CRenderer;

	// ----------------------------------------------------------------------------

	enum ShaderType
	{
		stVertexShader,
		stPixelShader
	};

	// ----------------------------------------------------------------------------

	class CShader
	{
		friend class CRenderer;

	private:
		static int referenceCounter;
		bool exists;

		CGprogram cgProgram;
		static CGparameter cgParameter;

		ShaderType shaderType;

	public:
		CShader() { exists = false; }



		void init(std::string fileName, ShaderType shaderType, int argsNum = 0, const char **args = NULL);
		void free();

		ShaderType getShaderType() const;



		static void setVertexShaderFloatConstant(const char *name, float x);

		static void setVertexShaderVectorConstant(const char *name, const CVector2 &v, float z, float w);
		static void setVertexShaderVectorConstant(const char *name, const CVector3 &v, float w);
		static void setVertexShaderVectorConstant(const char *name, const CVector4 &v);
		static void setVertexShaderVectorConstant(const char *name, float x, float y, float z, float w);

		static void setVertexShaderVectorArrayConstants(const char *name, CVector2 *v, float *z, float *w, int count);
		static void setVertexShaderVectorArrayConstants(const char *name, CVector3 *v, float *w, int count);
		static void setVertexShaderVectorArrayConstants(const char *name, CVector4 *v, int count);
		static void setVertexShaderVectorArrayConstants(const char *name, float *x, float *y, float *z, float *w, int count);

		static void setVertexShaderMatrixConstant(const char *name, const CMatrix &m);
		static void setVertexShaderMatrixArrayConstants(const char *name, CMatrix *m, int count);



		static void setPixelShaderFloatConstant(const char *name, float x);

		static void setPixelShaderVectorConstant(const char *name, const CVector2 &v, float z, float w);
		static void setPixelShaderVectorConstant(const char *name, const CVector3 &v, float w);
		static void setPixelShaderVectorConstant(const char *name, const CVector4 &v);
		static void setPixelShaderVectorConstant(const char *name, float x, float y, float z, float w);

		static void setPixelShaderVectorArrayConstants(const char *name, CVector2 *v, float *z, float *w, int count);
		static void setPixelShaderVectorArrayConstants(const char *name, CVector3 *v, float *w, int count);
		static void setPixelShaderVectorArrayConstants(const char *name, CVector4 *v, int count);
		static void setPixelShaderVectorArrayConstants(const char *name, float *x, float *y, float *z, float *w, int count);

		static void setPixelShaderMatrixConstant(const char *name, const CMatrix &m);
		static void setPixelShaderMatrixArrayConstants(const char *name, CMatrix *m, int count);
	};
}

// ----------------------------------------------------------------------------

#endif
