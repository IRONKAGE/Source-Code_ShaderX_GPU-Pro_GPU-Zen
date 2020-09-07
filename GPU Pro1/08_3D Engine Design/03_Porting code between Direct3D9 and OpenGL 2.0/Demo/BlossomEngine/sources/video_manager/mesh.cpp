/* $Id: mesh.cpp 277 2009-09-10 20:22:34Z maxest $ */

#ifdef RNDR_D3D
	#include <d3dx9.h>
#endif

#include <stdio.h>
#include <cmath>

#include "../renderer/vertex_declaration.h"
#include "../renderer/vertex_buffer.h"
#include "../renderer/index_buffer.h"
#include "../renderer/renderer.h"
#include "../renderer/shader.h"
#include "mesh.h"
#include "../math/blossom_engine_math.h"
#include "../common/blossom_engine_common.h"

// ----------------------------------------------------------------------------

namespace Blossom
{
	struct TempVertex
	{
		CVector3 position, normal, tangent, binormal;
		CVector2 texCoord0;
	};

	void CMesh::computeTangents(bool computeNewNormals)
	{
		#ifdef RNDR_D3D
		{
			D3DVERTEXELEMENT9 tempMeshVertexDeclaration[] =
			{
				{ 0, 0, D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0 },
				{ 0, 12, D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_NORMAL, 0 },
				{ 0, 24, D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TANGENT, 0 },
				{ 0, 36, D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_BINORMAL, 0 },
				{ 0, 48, D3DDECLTYPE_FLOAT2, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 0 },
				D3DDECL_END()
			};

			ID3DXMesh *tempMesh;
			D3DXCreateMesh(facesNum, 3*facesNum, D3DXMESH_SYSTEMMEM, tempMeshVertexDeclaration, CRenderer::D3DDevice, &tempMesh);

			TempVertex *tempVertices;
			WORD *tempIndices;

			tempMesh->LockVertexBuffer(0, (void**)&tempVertices);
			tempMesh->LockIndexBuffer(0, (void**)&tempIndices);
			{
				for (int i = 0; i < 3*facesNum; i++)
				{
					tempVertices[i].position = vertices[i].position;
					tempVertices[i].normal = vertices[i].normal;
					tempVertices[i].tangent = vertices[i].tangent;
					tempVertices[i].binormal = vertices[i].bitangent;
					tempVertices[i].texCoord0 = vertices[i].texCoord0;

					tempIndices[i] = i;
				}
			}
			tempMesh->UnlockIndexBuffer();
			{
				DWORD *adjacency = new DWORD[3*facesNum];
				tempMesh->GenerateAdjacency(epsilon, adjacency);

				D3DXComputeTangentFrameEx(tempMesh,
					D3DDECLUSAGE_TEXCOORD, 0,
					D3DDECLUSAGE_TANGENT, 0, D3DDECLUSAGE_BINORMAL, 0,
					D3DDECLUSAGE_NORMAL, 0,
					D3DXTANGENT_GENERATE_IN_PLACE | (computeNewNormals ? D3DXTANGENT_CALCULATE_NORMALS : 0),
					adjacency, 0.01f, 0.25f, 0.01f, NULL, NULL);

				delete[] adjacency;
			}
			for (int i = 0; i < 3*facesNum; i++)
			{
				vertices[i].tangent = tempVertices[i].tangent;
				vertices[i].bitangent = tempVertices[i].binormal;	
			}
			if (computeNewNormals)
			{
				for (int i = 0; i < 3*facesNum; i++)
					vertices[i].normal = tempVertices[i].normal;
			}
			tempMesh->UnlockVertexBuffer();

			tempMesh->Release();
		}
		#else
		{
			// generate tangent-basis using D3D renderer, save to Blossom mesh file format, and then you can use tangents with OpenGL renderer
		}
		#endif
	}



	void CMesh::computeFaceNormals()
	{
		for (int i = 0; i < facesNum; i++)
		{
			CVector3 v1 = vertices[indices[3*i + 1]].position - vertices[indices[3*i + 0]].position;
			CVector3 v2 = vertices[indices[3*i + 2]].position - vertices[indices[3*i + 0]].position;
			faceNormals[i] = (v1 ^ v2).getNormalized();
		}
	}



	void CMesh::computeBoundingBoxes()
	{
		for (int i = 0; i < framesNum; i++)
		{
			boundingBoxes[i].min = CVector3(1000000000.0f, 1000000000.0f, 1000000000.0f);
			boundingBoxes[i].max = CVector3(-1000000000.0f, -1000000000.0f, -1000000000.0f);

			for (int j = 0; j < 3*facesNum; j++)
			{
				CVector3 tempVertex = CVector4(vertices[indices[j]].position.x, vertices[indices[j]].position.y, vertices[indices[j]].position.z, 1.0f) * animationMatrices[i][(int)vertices[indices[j]].nodeIndex];

				if (tempVertex.x < boundingBoxes[i].min.x)
					boundingBoxes[i].min.x = tempVertex.x;
				if (tempVertex.y < boundingBoxes[i].min.y)
					boundingBoxes[i].min.y = tempVertex.y;
				if (tempVertex.z < boundingBoxes[i].min.z)
					boundingBoxes[i].min.z = tempVertex.z;

				if (tempVertex.x > boundingBoxes[i].max.x)
					boundingBoxes[i].max.x = tempVertex.x;
				if (tempVertex.y > boundingBoxes[i].max.y)
					boundingBoxes[i].max.y = tempVertex.y;
				if (tempVertex.z > boundingBoxes[i].max.z)
					boundingBoxes[i].max.z = tempVertex.z;
			}
		}
	}



	void CMesh::initVertexBuffer(int verticesNum)
	{
		vertexBuffer.init(verticesNum*sizeof(MeshVertex));
	}



	void CMesh::updateVertexBufferData()
	{
		MeshVertex *vertices;

		vertexBuffer.map((void*&)vertices);
			memcpy(vertices, this->vertices, vertexBuffer.getSize());
		vertexBuffer.unmap();
	}



	void CMesh::freeVertexBuffer()
	{
		vertexBuffer.free();
	}
	


	void CMesh::initIndexBuffer(int indicesNum)
	{
		indexBuffer.init(indicesNum*sizeof(unsigned short));
	}



	void CMesh::updateIndexBufferData()
	{
		unsigned short *indices;

		indexBuffer.map((void*&)indices);
			memcpy(indices, this->indices, indexBuffer.getSize());
		indexBuffer.unmap();
	}



	void CMesh::freeIndexBuffer()
	{
		indexBuffer.free();
	}



	bool CMesh::areVerticesEqual(const MeshVertex &vertex1, const MeshVertex &vertex2)
	{
		if (vertex1.position == vertex2.position &&
			vertex1.normal == vertex2.normal &&
			vertex1.tangent == vertex2.tangent &&
			vertex1.bitangent == vertex2.bitangent &&
			vertex1.texCoord0 == vertex2.texCoord0 &&
			fabs(vertex1.nodeIndex - vertex2.nodeIndex) < epsilon)
		{
			return true;
		}
		else
		{
			return false;
		}
	}



	void CMesh::init()
	{
		nodesNum = 0;
		verticesNum = 0;
		facesNum = 0;
		framesNum = 0;

		vertices = NULL;
		indices = NULL;
		faceNormals = NULL;
		animationMatrices = NULL;
		pivots = NULL;
		boundingBoxes = NULL;
	}



	void CMesh::free()
	{
		freeVerticesData();
		freeIndicesData();
		freeFaceNormalsData();
		freeAnimationData();
		freePivotsData();
		freeBoundingBoxesData();

		freeVertexBuffer();
		freeIndexBuffer();
	}



	void CMesh::freeVerticesData()
	{
		if (vertices != NULL)
		{
			delete[] vertices;
			vertices = NULL;
		}
	}



	void CMesh::freeIndicesData()
	{
		if (indices != NULL)
		{
			delete[] indices;
			indices = NULL;
		}
	}



	void CMesh::freeFaceNormalsData()
	{
		if (faceNormals != NULL)
		{
			delete[] faceNormals;
			faceNormals = NULL;
		}
	}



	void CMesh::freeAnimationData()
	{
		if (animationMatrices != NULL)
		{
			for (int i = 0; i < framesNum; i++)
				delete[] animationMatrices[i];
			delete[] animationMatrices;
			animationMatrices = NULL;
		}
	}



	void CMesh::freePivotsData()
	{
		if (pivots != NULL)
		{
			delete[] pivots;
			pivots = NULL;
		}
	}



	void CMesh::freeBoundingBoxesData()
	{
		if (boundingBoxes != NULL)
		{
			delete[] boundingBoxes;
			boundingBoxes = NULL;
		}
	}



	void CMesh::initData(int nodesNum, int verticesNum, int facesNum, int framesNum)
	{
		this->nodesNum = nodesNum;
		this->verticesNum = verticesNum;
		this->facesNum = facesNum;
		this->framesNum = framesNum;

		this->vertices = new MeshVertex[verticesNum];
		this->indices = new unsigned short[3*facesNum];
		this->faceNormals = new CVector3[facesNum];
		this->animationMatrices = new CMatrix*[framesNum];
		for (int i = 0; i < framesNum; i++)
			this->animationMatrices[i] = new CMatrix[nodesNum];
		this->boundingBoxes = new CBoundingBox[framesNum];

		initVertexBuffer(verticesNum);
		initIndexBuffer(3*facesNum);
	}



	void CMesh::updateData(const MeshVertex *vertices, const unsigned short *indices, bool computeTangents, bool computeFaceNormals, bool computeBoundingBoxes)
	{
		memcpy((void*)this->vertices, (void*)vertices, verticesNum*sizeof(MeshVertex));
		if (indices != NULL)
		{
			memcpy((void*)this->indices, (void*)indices, 3*facesNum*sizeof(unsigned short));
		}
		else
		{
			for (int i = 0; i < 3*facesNum; i++)
				this->indices[i] = i;
		}

		if (computeTangents)
			this->computeTangents(true);
		if (computeFaceNormals)
			this->computeFaceNormals();
		if (computeBoundingBoxes)
			this->computeBoundingBoxes();

		updateVertexBufferData();
		updateIndexBufferData();
	}



	void CMesh::importASE(std::string fileName, bool getNormalsFromFile)
	{
		int ticksPerFrame;
		int currentNode;
		MeshNode *nodes;
		int currentVertexNormalIndex = 0, currentVertexIndex = 0;
		char textLine[255], previousTextLine[255];
		char t[100];
		float f1, f2, f3, f4;
		int i1, i2, i3, i4;
		CVector3 nodeNormalTransform[3];
		CMatrix *nodeTranslationMatrices;



		FILE *file = fopen(fileName.c_str(), "rt");

		if (file == NULL)
		{
			CLogger::addText("\tERROR: couldn't import ASE file, %s\n", fileName.c_str());
			exit(1);
		}



		// counting nodes num
		while (!feof(file))
		{
			fgets(textLine, 255, file);
			
			if (strstr(textLine, "GEOMOBJECT"))
				this->nodesNum++;

			if (strstr(textLine, "SCENE_LASTFRAME"))
			{
				sscanf(textLine, "\t*SCENE_LASTFRAME %d", &this->framesNum);
				if (this->framesNum != 1) // for animated meshes we need one more frame for bind pose
					this->framesNum++;
			}

			if (strstr(textLine, "SCENE_TICKSPERFRAME"))
				sscanf(textLine, "\t*SCENE_TICKSPERFRAME %d", &ticksPerFrame);
		}



		nodes = new MeshNode[this->nodesNum];
		for (int i = 0; i < this->nodesNum; i++)
			nodes[i].parent = NULL;

		this->pivots = new CVector3[this->nodesNum];

		currentNode = -1;
		fseek(file, 0, SEEK_SET);



		// getting general nodes information
		while (!feof(file))
		{
			strcpy(previousTextLine, textLine);
			fgets(textLine, 255, file);
			
			if (strstr(textLine, "GEOMOBJECT"))
				currentNode++;

			if (strstr(textLine, "NODE_NAME") && strstr(previousTextLine, "GEOMOBJECT"))
			{
				sscanf(textLine, "\t*NODE_NAME %s", t);
				nodes[currentNode].name = t;
			}

			if (strstr(textLine, "NODE_PARENT") && strstr(previousTextLine, "NODE_NAME"))
			{
				sscanf(textLine, "\t*NODE_PARENT %s", t);
				for (int i = 0; i < currentNode+1; i++)
				{
					if (strcmp(nodes[i].name.c_str(), t) == 0)
						nodes[currentNode].parent = &nodes[i];
				}
			}

			if (strstr(textLine, "TM_ROW0"))
			{
				sscanf(textLine, "\t\t*TM_ROW0 %f\t%f\t%f", &f1, &f2, &f3);
				nodeNormalTransform[0] = CVector3(f1, f2, f3).getNormalized();
			}
			if (strstr(textLine, "TM_ROW1"))
			{
				sscanf(textLine, "\t\t*TM_ROW1 %f\t%f\t%f", &f1, &f2, &f3);
				nodeNormalTransform[1] = CVector3(f1, f2, f3).getNormalized();
			}
			if (strstr(textLine, "TM_ROW2"))
			{
				sscanf(textLine, "\t\t*TM_ROW2 %f\t%f\t%f", &f1, &f2, &f3);
				nodeNormalTransform[2] = CVector3(f1, f2, f3).getNormalized();

				nodes[currentNode].normalTransform = CMatrix(
					nodeNormalTransform[0].x, nodeNormalTransform[0].y, nodeNormalTransform[0].z, 0.0f,
					nodeNormalTransform[1].x, nodeNormalTransform[1].y, nodeNormalTransform[1].z, 0.0f,
					nodeNormalTransform[2].x, nodeNormalTransform[2].y, nodeNormalTransform[2].z, 0.0f,
					0.0f					, 0.0f					  , 0.0f					, 1.0f);
			}

			if (strstr(textLine, "TM_POS"))
			{
				sscanf(textLine, "\t\t*TM_POS %f\t%f\t%f", &f1, &f2, &f3);
				nodes[currentNode].pivot = CVector4(f1, f2, f3);
				this->pivots[currentNode] = nodes[currentNode].pivot;
			}

			if (strstr(textLine, "MESH_NUMVERTEX"))
			{
				sscanf(textLine, "\t\t*MESH_NUMVERTEX %d", &i1);
				nodes[currentNode].verticesNum = i1;
			}

			if (strstr(textLine, "MESH_NUMFACES"))
			{
				sscanf(textLine, "\t\t*MESH_NUMFACES %d", &i1);
				nodes[currentNode].facesNum = i1;
				this->facesNum += i1;
			}

			if (strstr(textLine, "MESH_NUMTVERTEX"))
			{
				sscanf(textLine, "\t\t*MESH_NUMTVERTEX %d", &i1);
				nodes[currentNode].texCoordsNum = i1;

				nodes[currentNode].vertices = new CVector3[nodes[currentNode].verticesNum];
				nodes[currentNode].texCoords = new CVector2[nodes[currentNode].texCoordsNum];
				nodes[currentNode].faces = new NodeFace[nodes[currentNode].facesNum];
				nodes[currentNode].verticesNormals = new CVector3[3*nodes[currentNode].facesNum];
				nodes[currentNode].facesNormals = new CVector3[nodes[currentNode].facesNum];
				nodes[currentNode].translationMatrices = new CMatrix[this->framesNum];
				nodes[currentNode].rotationMatrices = new CMatrix[this->framesNum];
				nodes[currentNode].animationMatrices = new CMatrix[this->framesNum];
			}
		}



		nodeTranslationMatrices = new CMatrix[this->framesNum];

		currentNode = -1;
		fseek(file, 0, SEEK_SET);



		// getting vertices data
		while (!feof(file))
		{
			strcpy(previousTextLine, textLine);
			fgets(textLine, 255, file);

			if (strstr(textLine, "GEOMOBJECT"))
			{
				currentNode++;
				currentVertexNormalIndex = 0;
			}

			if (strstr(textLine, "MESH_VERTEX "))
			{
				sscanf(textLine, "\t\t\t*MESH_VERTEX    %d\t%f\t%f\t%f", &i1, &f1, &f2, &f3);
				nodes[currentNode].vertices[i1] = CVector3(f1, f2, f3);
			}

			if (strstr(textLine, "MESH_FACE "))
			{
				sscanf(textLine, "\t\t\t*MESH_FACE    %d:    A:    %d B:    %d C:    %d", &i1, &i2, &i3, &i4);
				nodes[currentNode].faces[i1].vertexIndex[0] = i2;
				nodes[currentNode].faces[i1].vertexIndex[1] = i3;
				nodes[currentNode].faces[i1].vertexIndex[2] = i4;
			}

			if (strstr(textLine, "MESH_TVERT "))
			{
				sscanf(textLine, "\t\t\t*MESH_TVERT %d\t%f\t%f", &i1, &f1, &f2);
				nodes[currentNode].texCoords[i1] = CVector2(f1, f2);
			}

			if (strstr(textLine, "MESH_TFACE "))
			{
				sscanf(textLine, "\t\t\t*MESH_TFACE %d\t%d\t%d\t%d", &i1, &i2, &i3, &i4);
				nodes[currentNode].faces[i1].texCoordIndex[0] = i2;
				nodes[currentNode].faces[i1].texCoordIndex[1] = i3;
				nodes[currentNode].faces[i1].texCoordIndex[2] = i4;
			}

			if (strstr(textLine, "MESH_FACENORMAL "))
			{
				sscanf(textLine, "\t\t\t*MESH_FACENORMAL %d\t%f\t%f\t%f", &i1, &f1, &f2, &f3);
				nodes[currentNode].facesNormals[i1] = CVector3(f1, f2, f3);
			}

			if (strstr(textLine, "MESH_VERTEXNORMAL "))
			{
				sscanf(textLine, "\t\t\t\t*MESH_VERTEXNORMAL %d\t%f\t%f\t%f", &i1, &f1, &f2, &f3);
				nodes[currentNode].verticesNormals[currentVertexNormalIndex] = CVector3(f1, f2, f3);
				currentVertexNormalIndex++;
			}
	
			if (strstr(textLine, "CONTROL_POS_SAMPLE"))
			{
				sscanf(textLine, "\t\t\t*CONTROL_POS_SAMPLE %d\t%f\t%f\t%f", &i1, &f1, &f2, &f3);
				if (i1 != 0) // transformation of the first frame does not concern us
				{
					if (nodes[currentNode].parent == NULL)
						nodeTranslationMatrices[i1/ticksPerFrame] = CMatrix::translate(f1 - nodes[currentNode].pivot.x, f2 - nodes[currentNode].pivot.y, f3 - nodes[currentNode].pivot.z);
					else
						nodeTranslationMatrices[i1/ticksPerFrame] = CMatrix::translate(f1 - nodes[currentNode].pivot.x + nodes[currentNode].parent->pivot.x, f2 - nodes[currentNode].pivot.y + nodes[currentNode].parent->pivot.y, f3 - nodes[currentNode].pivot.z + nodes[currentNode].parent->pivot.z);
				
					nodes[currentNode].translationMatrices[i1/ticksPerFrame] = nodeTranslationMatrices[i1/ticksPerFrame]; // * nodeTranslationMatrices[i1/ticksPerFrame - 1].getInversed();
					for (int i = 0; i < 3; i++)
						nodes[currentNode].translationMatrices[i1/ticksPerFrame](3, i) -= nodeTranslationMatrices[i1/ticksPerFrame - 1](3, i);
				}
			}
			if (strstr(textLine, "CONTROL_ROT_SAMPLE"))
			{
				sscanf(textLine, "\t\t\t*CONTROL_ROT_SAMPLE %d\t%f\t%f\t%f\t%f", &i1, &f1, &f2, &f3, &f4);
				if (i1 != 0) // transformation of the first frame does not concern us
				{
					nodes[currentNode].rotationMatrices[i1/ticksPerFrame] = CMatrix::rotate(-f4, f1, f2, f3);
				}
			}
		}



		delete[] nodeTranslationMatrices;



		fclose(file);



		this->vertices = new MeshVertex[3*this->facesNum];
		this->indices = new unsigned short[3*this->facesNum];
		this->faceNormals = new CVector3[this->facesNum];
		this->animationMatrices = new CMatrix*[this->framesNum];
		for (int i = 0; i < this->framesNum; i++)
			this->animationMatrices[i] = new CMatrix[this->nodesNum];

		this->boundingBoxes = new CBoundingBox[this->framesNum];

		for (int i = 0; i < this->nodesNum; i++)
		{
			for (int j = 0; j < nodes[i].facesNum; j++)
			{
				this->vertices[currentVertexIndex + 0].position = nodes[i].vertices[nodes[i].faces[j].vertexIndex[0]];
				this->vertices[currentVertexIndex + 0].normal = nodes[i].verticesNormals[3*j + 0] * nodes[i].normalTransform;
				this->vertices[currentVertexIndex + 0].tangent = CVector3();
				this->vertices[currentVertexIndex + 0].bitangent = CVector3();
				this->vertices[currentVertexIndex + 0].texCoord0 = nodes[i].texCoords[nodes[i].faces[j].texCoordIndex[0]];
				this->vertices[currentVertexIndex + 0].nodeIndex = (float)i + 0.1f; // we make sure the value is above "i" so in a shader truncation can work properly

				this->vertices[currentVertexIndex + 1].position = nodes[i].vertices[nodes[i].faces[j].vertexIndex[1]];
				this->vertices[currentVertexIndex + 1].normal = nodes[i].verticesNormals[3*j + 1] * nodes[i].normalTransform;
				this->vertices[currentVertexIndex + 1].tangent = CVector3();
				this->vertices[currentVertexIndex + 1].bitangent = CVector3();
				this->vertices[currentVertexIndex + 1].texCoord0 = nodes[i].texCoords[nodes[i].faces[j].texCoordIndex[1]];
				this->vertices[currentVertexIndex + 1].nodeIndex = (float)i + 0.1f; // we make sure the value is above "i" so in a shader truncation can work properly

				this->vertices[currentVertexIndex + 2].position = nodes[i].vertices[nodes[i].faces[j].vertexIndex[2]];
				this->vertices[currentVertexIndex + 2].normal = nodes[i].verticesNormals[3*j + 2] * nodes[i].normalTransform;
				this->vertices[currentVertexIndex + 2].tangent = CVector3();
				this->vertices[currentVertexIndex + 2].bitangent = CVector3();
				this->vertices[currentVertexIndex + 2].texCoord0 = nodes[i].texCoords[nodes[i].faces[j].texCoordIndex[2]];
				this->vertices[currentVertexIndex + 2].nodeIndex = (float)i + 0.1f; // we make sure the value is above "i" so in a shader truncation can work properly

				this->faceNormals[currentVertexIndex/3] = nodes[i].facesNormals[j];

				currentVertexIndex += 3;
			}
		}

		computeTangents(!getNormalsFromFile);

		// these arrays are to help to compute indices for vertices
		bool *checkedVertices = new bool[3*this->facesNum];
		for (int i = 0; i < 3*this->facesNum; i++)
			checkedVertices[i] = false;
		std::vector<MeshVertex> indexedVertices; // here, indexed vertices will be stored, and finnaly coppied to this->vertices

		for (int i = 0; i < 3*facesNum; i++)
		{
			if (!checkedVertices[i])
			{
				checkedVertices[i] = true;
				indexedVertices.push_back(this->vertices[i]);
				this->indices[i] = indexedVertices.size() - 1;

				for (int j = i+1; j < 3*facesNum; j++)
				{
					if (!checkedVertices[j])
					{
						if (areVerticesEqual(this->vertices[i], this->vertices[j]))
						{
							checkedVertices[j] = true;
							this->indices[j] = indexedVertices.size() - 1;
						}
					}
				}
			}
		}

		delete[] this->vertices;

		this->verticesNum = indexedVertices.size();
		this->vertices = new MeshVertex[this->verticesNum];

		for (int i = 0; i < this->verticesNum; i++)
		{
			this->vertices[i] = indexedVertices[i];
		}

		delete[] checkedVertices;



		for (int i = 0; i < this->nodesNum; i++)
		{
			for (int j = 1; j < this->framesNum; j++)
			{
				nodes[i].rotationMatrices[j] = nodes[i].rotationMatrices[j-1] * nodes[i].rotationMatrices[j];
				nodes[i].translationMatrices[j] = nodes[i].translationMatrices[j-1] * nodes[i].translationMatrices[j];
			}
		}
		// here we make an assumption that every child's parent is processed before the child
		for (int i = 0; i < this->nodesNum; i++)
		{
			for (int j = 0; j < this->framesNum; j++)
			{
				if (nodes[i].parent == NULL)
					nodes[i].animationMatrices[j] = CMatrix::translate(-nodes[i].pivot.x, -nodes[i].pivot.y, -nodes[i].pivot.z) *
													nodes[i].rotationMatrices[j] *
													CMatrix::translate(nodes[i].pivot.x, nodes[i].pivot.y, nodes[i].pivot.z) *
													nodes[i].translationMatrices[j];
				else
					nodes[i].animationMatrices[j] = CMatrix::translate(-nodes[i].pivot.x, -nodes[i].pivot.y, -nodes[i].pivot.z) *
													nodes[i].parent->normalTransform.getInversed() *
													nodes[i].rotationMatrices[j] *
													nodes[i].parent->normalTransform *
													CMatrix::translate(nodes[i].pivot.x, nodes[i].pivot.y, nodes[i].pivot.z) *
													nodes[i].translationMatrices[j] *
													nodes[i].parent->animationMatrices[j];
			}
		}
		for (int i = 0; i < this->nodesNum; i++)
		{
			for (int j = 0; j < this->framesNum; j++)
			{
				this->animationMatrices[j][i] = nodes[i].animationMatrices[j];
			}
		}



		for (int i = 0; i < this->nodesNum; i++)
		{
			delete[] nodes[i].vertices;
			delete[] nodes[i].texCoords;
			delete[] nodes[i].faces;
			delete[] nodes[i].verticesNormals;
			delete[] nodes[i].facesNormals;
			delete[] nodes[i].translationMatrices;
			delete[] nodes[i].rotationMatrices;
			delete[] nodes[i].animationMatrices;
		}
		delete[] nodes;



		computeFaceNormals();
		computeBoundingBoxes();

		initVertexBuffer(verticesNum);
		updateVertexBufferData();
		initIndexBuffer(3*facesNum);
		updateIndexBufferData();



		CLogger::addText("\tOK: ASE file imported, %s", fileName.c_str());
	}



	void CMesh::loadDataFromFile(std::string fileName)
	{
		FILE *file = fopen(fileName.c_str(), "rb");

		if (file == NULL)
		{
			CLogger::addText("\tERROR: couldn't load mesh from file, %s\n", fileName.c_str());
			exit(1);
		}

		fread((void*)&nodesNum, 4, 1, file);
		fread((void*)&verticesNum, 4, 1, file);
		fread((void*)&facesNum, 4, 1, file);
		fread((void*)&framesNum, 4, 1, file);

		vertices = new MeshVertex[3*facesNum];
		indices = new unsigned short[3*facesNum];
		faceNormals = new CVector3[facesNum];
		animationMatrices = new CMatrix*[framesNum];
		for (int i = 0; i < framesNum; i++)
			animationMatrices[i] = new CMatrix[nodesNum];
		pivots = new CVector3[nodesNum];
		boundingBoxes = new CBoundingBox[framesNum];

		fread((void*)vertices, sizeof(MeshVertex), verticesNum, file);
		fread((void*)indices, sizeof(unsigned short), 3*facesNum, file);
		fread((void*)faceNormals, sizeof(CVector3), facesNum, file);
		for (int i = 0; i < framesNum; i++)
			fread((void*)animationMatrices[i], sizeof(CMatrix), nodesNum, file);
		fread((void*)pivots, sizeof(CVector3), nodesNum, file);
		fread((void*)boundingBoxes, sizeof(CBoundingBox), framesNum, file);

		initVertexBuffer(verticesNum);
		updateVertexBufferData();
		initIndexBuffer(3*facesNum);
		updateIndexBufferData();

		fclose(file);

		CLogger::addText("\tOK: mesh loaded from file, %s\n", fileName.c_str());
	}



	void CMesh::saveDataToFile(std::string fileName)
	{
		FILE *file = fopen(fileName.c_str(), "wb");

		if (file == NULL)
		{
			CLogger::addText("\tERROR: couldn't save mesh to file, %s\n", fileName.c_str());
			exit(1);
		}

		fwrite((const void*)&nodesNum, 4, 1, file);
		fwrite((const void*)&verticesNum, 4, 1, file);
		fwrite((const void*)&facesNum, 4, 1, file);
		fwrite((const void*)&framesNum, 4, 1, file);

		fwrite((const void*)vertices, sizeof(MeshVertex), verticesNum, file);
		fwrite((const void*)indices, sizeof(unsigned short), 3*facesNum, file);
		fwrite((const void*)faceNormals, sizeof(CVector3), facesNum, file);
		for (int i = 0; i < framesNum; i++)
			fwrite((const void*)animationMatrices[i], sizeof(CMatrix), nodesNum, file);
		fwrite((const void*)pivots, sizeof(CVector3), nodesNum, file);
		fwrite((const void*)boundingBoxes, sizeof(CBoundingBox), framesNum, file);

		fclose(file);

		CLogger::addText("\tOK: mesh saved to file, %s\n", fileName.c_str());
	}



	int CMesh::getNodesNum() const
	{
		return nodesNum;
	}



	int CMesh::getVerticesNum() const
	{
		return verticesNum;
	}



	int CMesh::getFacesNum() const
	{
		return facesNum;
	}



	int CMesh::getFramesNum() const
	{
		return framesNum;
	}



	CBoundingBox CMesh::getBoundingBox(const CMatrix &worldTransform, int frame) const
	{
		CBoundingBox boundingBox;

		boundingBox = boundingBoxes[frame];
		boundingBox.transform(worldTransform);

		return boundingBox;
	}



	CBoundingBox CMesh::getBoundingBox(const CMatrix &worldTransform, int frame1, int frame2, float interpolationProgress) const
	{
		CBoundingBox boundingBox;

		CVector3 tempCorners1[8];
		CVector3 tempCorners2[8];
		CVector3 tempCorners3[8];

		boundingBoxes[frame1].getCorners(tempCorners1);
		boundingBoxes[frame2].getCorners(tempCorners2);

		for (int i = 0; i < 8; i++)
		{
			tempCorners1[i] = CVector4(tempCorners1[i].x, tempCorners1[i].y, tempCorners1[i].z, 1.0f) * worldTransform;
			tempCorners2[i] = CVector4(tempCorners2[i].x, tempCorners2[i].y, tempCorners2[i].z, 1.0f) * worldTransform;
			tempCorners3[i] = tempCorners1[i]*(1.0f-interpolationProgress) + tempCorners2[i]*interpolationProgress;
		}

		boundingBox.updateWithCorners(tempCorners3);

		return boundingBox;
	}



	void CMesh::render()
	{
		CRenderer::setVertexDeclaration(CRenderer::meshVertexDeclaration);
		CRenderer::setVertexBuffer(vertexBuffer);
		CRenderer::setIndexBuffer(indexBuffer);

		CRenderer::drawIndexedPrimitives(ptTriangleList, verticesNum, facesNum);
	}



	void CMesh::render(const CMatrix &worldTransform, const CMatrix &viewProjTransform)
	{
		CRenderer::setVertexDeclaration(CRenderer::meshVertexDeclaration);
		CRenderer::setVertexBuffer(vertexBuffer);
		CRenderer::setIndexBuffer(indexBuffer);
		CRenderer::setShader(CRenderer::meshStaticVertexShader);
		{
			CShader::setVertexShaderMatrixConstant("worldTransform", worldTransform);
			CShader::setVertexShaderMatrixConstant("worldTransformInversed", worldTransform.getInversed());
			CShader::setVertexShaderMatrixConstant("viewProjTransform", viewProjTransform);
		}
		CRenderer::setShader(CRenderer::meshPixelShader);

		CRenderer::drawIndexedPrimitives(ptTriangleList, verticesNum, facesNum);
	}



	void CMesh::render(const CMatrix &worldTransform, const CMatrix &viewProjTransform, int frame)
	{
		CRenderer::setVertexDeclaration(CRenderer::meshVertexDeclaration);
		CRenderer::setVertexBuffer(vertexBuffer);
		CRenderer::setIndexBuffer(indexBuffer);
		CRenderer::setShader(CRenderer::meshAnimationVertexShader);
		{
			CShader::setVertexShaderMatrixConstant("worldTransform", worldTransform);
			CShader::setVertexShaderMatrixConstant("worldTransformInversed", worldTransform.getInversed());
			CShader::setVertexShaderMatrixArrayConstants("animationMatrices", animationMatrices[frame], nodesNum);
			CShader::setVertexShaderMatrixConstant("viewProjTransform", viewProjTransform);
		}
		CRenderer::setShader(CRenderer::meshPixelShader);

		CRenderer::drawIndexedPrimitives(ptTriangleList, verticesNum, facesNum);
	}



	void CMesh::render(const CMatrix &worldTransform, const CMatrix &viewProjTransform, int frame1, int frame2, float interpolationProgress)
	{
		CRenderer::setVertexDeclaration(CRenderer::meshVertexDeclaration);
		CRenderer::setVertexBuffer(vertexBuffer);
		CRenderer::setIndexBuffer(indexBuffer);
		CRenderer::setShader(CRenderer::meshAnimationInterpolationVertexShader);
		{
			CShader::setVertexShaderMatrixConstant("worldTransform", worldTransform);
			CShader::setVertexShaderMatrixConstant("worldTransformInversed", worldTransform.getInversed());
			CShader::setVertexShaderMatrixArrayConstants("animationMatrices1", animationMatrices[frame1], nodesNum);
			CShader::setVertexShaderMatrixArrayConstants("animationMatrices2", animationMatrices[frame2], nodesNum);
			CShader::setVertexShaderMatrixConstant("viewProjTransform", viewProjTransform);
			CShader::setVertexShaderFloatConstant("interpolationProgress", interpolationProgress);
		}
		CRenderer::setShader(CRenderer::meshPixelShader);

		CRenderer::drawIndexedPrimitives(ptTriangleList, verticesNum, facesNum);
	}
}
