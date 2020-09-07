#include "dxstdafx.h"
#include "ShadedMesh.h"
#include "RenderContext.h"

ShadedMesh::ShadedMesh(LPD3DXMESH mesh, LPD3DXBUFFER materialBuffer, unsigned int nSubmeshes, TextureDirectory& textureDirectory, LPDIRECT3DDEVICE9 device)
{
	this->mesh = mesh;
	computeBoundingBox();

	roles[L"default"] = new Role(materialBuffer, nSubmeshes, textureDirectory, device);
}

ShadedMesh::ShadedMesh(LPD3DXMESH mesh)
{
	this->mesh = mesh;
	computeBoundingBox();
}

ShadedMesh::~ShadedMesh(void)
{
	RoleDirectory::iterator i = roles.begin();
	while(i != roles.end())
	{
		delete i->second;
		i++;
	}
}


void ShadedMesh::render(const RenderContext& context)
{
	RoleDirectory::iterator iRole = roles.find(context.roleName);
	if(iRole != roles.end())
		iRole->second->render(context, mesh);
}

void ShadedMesh::addRole(const UnicodeString& name, Role* role)
{
	roles[name] = role;
}

void ShadedMesh::computeBoundingBox()
{
	D3DXVECTOR3* pData; 
	
	mesh->LockVertexBuffer( 0, (LPVOID*) &pData ) ;

	D3DXComputeBoundingBox( pData,
							mesh->GetNumVertices(),
							D3DXGetFVFVertexSize(mesh->GetFVF()),
							&boundingBoxMin,
							&boundingBoxMax
	     				   );

	mesh->UnlockVertexBuffer();
}