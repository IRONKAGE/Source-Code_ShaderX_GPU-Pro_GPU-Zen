#include "dxstdafx.h"
#include "MeshObject.h"

MeshObject::MeshObject(LPD3DXMESH meshp,int index)
{
	mesh = meshp;
	mesh_index = index;
	translation = D3DXVECTOR3(0,0,0);

	boundingMin = D3DXVECTOR3(0,0,0);
	boundingMax = D3DXVECTOR3(0,0,0);
	submesh_number = 0;
}

MeshObject::~MeshObject()
{
}

void MeshObject::SetTranslation(D3DXVECTOR3 p)
{
	translation = p;
}

D3DXVECTOR3 MeshObject::GetTranslation()
{
	return translation;
}

LPD3DXMESH MeshObject::getMesh()
{
	return mesh;
}

int MeshObject::getMeshIndex()
{
	return mesh_index;
}

bool MeshObject::ComputeBoundingBox()
{
    HRESULT hr = 0;

    D3DXVECTOR3* pData; 
	
    mesh->LockVertexBuffer( 0, (LPVOID*) &pData ) ;

    hr = D3DXComputeBoundingBox( pData,
								 mesh->GetNumVertices(),
								 D3DXGetFVFVertexSize(mesh->GetFVF()),
								 &boundingMin,
								 &boundingMax
								);

    mesh->UnlockVertexBuffer();

    if( FAILED(hr) )
        return false;

    return true;
}

void MeshObject::SetBoundingBox(D3DXVECTOR3 bMin,D3DXVECTOR3 bMax)
{
	boundingMin = bMin;
	boundingMax = bMax;
}

void MeshObject::SetNumberOfSubMeshes(int n)
{
	submesh_number = n;
}

int MeshObject::GetNumberOfSubMeshes()
{
	return submesh_number;
}

float MeshObject::GetRadius()
{
	return max( D3DXVec3Length(&boundingMin), D3DXVec3Length(&boundingMax) );
}
