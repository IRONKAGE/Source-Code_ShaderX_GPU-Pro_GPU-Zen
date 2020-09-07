#pragma once
#include <map>

#include "UnicodeString.h"
#include "AnsiString.h"
class ShadedMesh;
class Entity;
class SpotLight;
class Role;
class RigidModel;

struct CompareStringsA
{
	bool operator() (AnsiString a, AnsiString b) const
	{
		return a < b;
	}
};

struct CompareStringsW
{
	bool operator() (UnicodeString a, UnicodeString b) const
	{
		return a < b;
	}
};

typedef std::map<const AnsiString, LPDIRECT3DTEXTURE9, CompareStringsA> TextureDirectory;
typedef std::map<const AnsiString, LPDIRECT3DCUBETEXTURE9, CompareStringsA> CubeTextureDirectory;
typedef std::map<const AnsiString, D3DXVECTOR4, CompareStringsA> VectorDirectory;

typedef std::map<const UnicodeString, LPD3DXMESH, CompareStringsW>     MeshDirectory;
typedef std::map<const UnicodeString, ShadedMesh*, CompareStringsW>	  ShadedMeshDirectory;
typedef std::map<const UnicodeString, Role*, CompareStringsW>	  RoleDirectory;
typedef std::map<const UnicodeString, Entity*, CompareStringsW>	  EntityDirectory;
typedef std::map<const UnicodeString, RigidModel*, CompareStringsW>	  RigidModelDirectory;
typedef std::map<const UnicodeString, SpotLight*, CompareStringsW>	  SpotLightDirectory;
	

