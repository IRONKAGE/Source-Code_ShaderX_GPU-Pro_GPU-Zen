#pragma once



class MeshObject
{
private:

	LPD3DXMESH mesh;

	int mesh_index;

	D3DXVECTOR3 translation;

	int submesh_number;

public:
	
	D3DXVECTOR3 boundingMin, boundingMax;
	
	MeshObject(LPD3DXMESH id, int index);
	void SetTranslation(D3DXVECTOR3 p);
	D3DXVECTOR3 GetTranslation();
	LPD3DXMESH getMesh();
	int getMeshIndex();
	bool ComputeBoundingBox();
	void SetBoundingBox(D3DXVECTOR3 bMin,D3DXVECTOR3 bMax);
	void SetNumberOfSubMeshes(int n);
	int GetNumberOfSubMeshes();
	float GetRadius();

	~MeshObject();


};
