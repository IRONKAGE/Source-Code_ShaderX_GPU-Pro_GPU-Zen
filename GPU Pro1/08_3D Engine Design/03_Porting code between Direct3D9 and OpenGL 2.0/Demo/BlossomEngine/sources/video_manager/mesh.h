/* $Id: mesh.h 183 2009-08-27 17:10:14Z maxest $ */

#ifndef _BLOSSOM_ENGINE_MESH_
#define _BLOSSOM_ENGINE_MESH_

// ----------------------------------------------------------------------------

namespace Blossom
{
	class CVector2;
	class CVector3;
	class CVector4;
	class CMatrix;
	class CBoundingBox;
	class CVertexDeclaration;
	class CVertexBuffer;
	class CIndexBuffer;
	class CShader;
	class CVideoManager;

	// ----------------------------------------------------------------------------

	struct MeshVertex
	{
		CVector3 position;
		CVector3 normal;
		CVector3 tangent;
		CVector3 bitangent; // that's right - it is NOT binormal!
		CVector2 texCoord0;
		float nodeIndex;
	};

	// ----------------------------------------------------------------------------

	class CMesh
	{
		friend class CVideoManager;

		// ----------------------------------------------------------------------------

		struct NodeFace
		{
			int vertexIndex[3];
			int texCoordIndex[3];
		};

		struct MeshNode
		{
			std::string name;
			MeshNode *parent;
			CMatrix normalTransform; // every loaded normal must be multiplied by this matrix
			CVector3 pivot; // pivot's position

			int verticesNum;
			int texCoordsNum;
			int facesNum;

			CVector3 *vertices;
			CVector2 *texCoords;
			NodeFace *faces;
			CVector3 *verticesNormals;
			CVector3 *facesNormals;

			CMatrix *translationMatrices;
			CMatrix *rotationMatrices;
			CMatrix *animationMatrices;
		};

		// ----------------------------------------------------------------------------

	private:
		int nodesNum;
		int verticesNum;
		int facesNum;
		int framesNum;

		CVertexBuffer vertexBuffer;
		CIndexBuffer indexBuffer;

		CBoundingBox *boundingBoxes;

	private:
		void computeTangents(bool computeNewNormals); // works only with Direct3D renderer!
		void computeFaceNormals();
		void computeBoundingBoxes();

		void initVertexBuffer(int verticesNum);
		void updateVertexBufferData();
		void freeVertexBuffer();

		void initIndexBuffer(int indicesNum);
		void updateIndexBufferData();
		void freeIndexBuffer();

		bool areVerticesEqual(const MeshVertex &vertex1, const MeshVertex &vertex2);

	public:
		MeshVertex *vertices;
		unsigned short *indices;
		CVector3 *faceNormals;
		CMatrix **animationMatrices;
		CVector3 *pivots;

	public:
		void init();
		void free();

		void freeVerticesData();
		void freeIndicesData();
		void freeFaceNormalsData();
		void freeAnimationData();
		void freePivotsData();
		void freeBoundingBoxesData();

		void initData(int nodesNum, int verticesNum, int facesNum, int framesNum);
		void updateData(const MeshVertex *vertices, const unsigned short *indices, bool computeTangents = false, bool computeFaceNormals = true, bool computeBoundingBoxes = true); // if indices are not passed (NULL), they will be generated (indices[n] = n)

		void importASE(std::string fileName, bool getNormalsFromFile = true);
		void loadDataFromFile(std::string fileName);
		void saveDataToFile(std::string fileName);

		int getNodesNum() const;
		int getVerticesNum() const;
		int getFacesNum() const;
		int getFramesNum() const;

		CBoundingBox getBoundingBox(const CMatrix &worldTransform, int frame = 0) const;
		CBoundingBox getBoundingBox(const CMatrix &worldTransform, int frame1, int frame2, float interpolationProgress) const;

		void render();
		void render(const CMatrix &worldTransform, const CMatrix &viewProjTransform);
		void render(const CMatrix &worldTransform, const CMatrix &viewProjTransform, int frame);
		void render(const CMatrix &worldTransform, const CMatrix &viewProjTransform, int frame1, int frame2, float interpolationProgress);
	};
}

// ----------------------------------------------------------------------------

#endif
