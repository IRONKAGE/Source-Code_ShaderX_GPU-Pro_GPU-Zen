#pragma once

#include "MeshObject.h"
#include "EngineInterface.h"

#define SNIPPET_SIZE 3.0f				//caustic quad size
#define  PHOTON_MAP_SIZE 32
#define  CausticsIntensity 0.012f
#define indexOfRefraction 0.98f			//IOR of the chess pieces

class Chess : public EngineInterface
{

	private:

		bool useClassicalMethod;		//if it is true, the generation of the environment maps is done without the separation of the dynamic environment

		static const int OBJECTS_SIZE = 34;		//number of objects(pieces, board)
		static const int MESH_SIZE = 8;			//number of meshes
		
		float fresnelFactor;					//the fresnel factor of the glass pieces

		D3DXMATRIX TexScaleBiasMatrix;			//transforms points from normalized device space to texture space

		//scene bounding box min and max coordinates
		D3DXVECTOR3 sceneBoundingBoxMin;
		D3DXVECTOR3 sceneBoundingBoxMax;
		
		D3DXVECTOR3 offset;		
		int moving_obj_index;			//the index of the moving object. If it equals 0, then all pieces are still.
		
		float IntensityOfSnippet[9];
		float SizeOfSnippet[9];
		
		//quads for caustic generation
		LPDIRECT3DVERTEXBUFFER9 causticQuadVertexBuffer;
		LPDIRECT3DINDEXBUFFER9 causticQuadIndexBuffer;
		LPDIRECT3DVERTEXBUFFER9 causticQuadInstanceBuffer;
		LPDIRECT3DVERTEXDECLARATION9 causticQuadVertexDecl;

		LPDIRECT3DTEXTURE9 SnippetTexture;			//texture for gaussian filtering during photon splatting

		enum { RENDER_REFRACTION_MAP, RENDER_ENVMAP_INIT, RENDER_ENVMAP, RENDER_ALL_ENVMAP, RENDER_JUST_MOVING_ENVMAP,
			   RENDER_ALL_LIGHTMAP, RENDER_JUST_MOVING_LIGHTMAP };

		D3DXVECTOR3 square_size;		//size of a square on the chessboard

		LPD3DXMESH mesh[MESH_SIZE];				//the array of the meshes
		MeshObject* objects[OBJECTS_SIZE];		//the aray of the objects
		D3DMATERIAL9* materials;				
		LPDIRECT3DTEXTURE9* textures;
					
		LPD3DXEFFECT effect;				
		
		

		D3DLIGHT9 light;					//the direct light
		CFirstPersonCamera camera;

		LPDIRECT3DSURFACE9 refractorSurface;
		LPDIRECT3DSURFACE9 environmentSurface;

		//environment distance impostors of the chess pieces. Aside from distance values, they store the incoming radiance.
		LPDIRECT3DCUBETEXTURE9 refractormap[MESH_SIZE-1];
		//the object distance impostors of the chess pieces. Aside from distance values, they store surface normals.
		LPDIRECT3DCUBETEXTURE9 environmentmap[OBJECTS_SIZE-1];

		//the height map impostors of the chess pieces. Aside from height values, they store surface normals.
		LPDIRECT3DTEXTURE9 HeightMap[MESH_SIZE-1];
		LPDIRECT3DSURFACE9 HeightMapStencilSurface;

		LPDIRECT3DSURFACE9 causticMapStencilSurface;
		LPDIRECT3DTEXTURE9 causticMap;				//the photon map storing photon hits in texture space

		LPDIRECT3DTEXTURE9 shadowMap;
		LPDIRECT3DSURFACE9 shadowMapStencilSurface;

		LPDIRECT3DTEXTURE9 lightMap;				//light map for the static environment
		LPDIRECT3DTEXTURE9 lightMap_moving_obj; //light map for the dynamic environment (contains the shadow and caustics of the moving piece)
		LPDIRECT3DSURFACE9 lightMapStencilSurface;

		LPDIRECT3DCUBETEXTURE9 SkyCubeMap;			//sky texture
		LPDIRECT3DVERTEXBUFFER9 full_screen_quad;	//full-screen quad
	
		ID3DXFont* Font;

		void RenderFullScreenQuad();
		void RenderGUI();
		bool GenerateHeightMap(int mesh_iterator);			//generates the height map impostor of the choosen piece 
		void CreateRefractorMap(D3DXMATRIX viewMatrix, D3DXMATRIX projMatrix, int mesh_iterator);	//generates the object distance impostor of the choosen piece
		//creates the environment distance impostor of the choosen piece
		void CreateEnvironmentMap(D3DXVECTOR3* Eye, D3DXMATRIX viewMatrix, D3DXMATRIX projMatrix, int obj_index, int method);
		bool RenderSceneIntoCubeMap(int iterator, int method);	//handles impostor rendering.		
		void CubeMapInitialization(bool render_refrmap, int render_envmap_options);	//Renders the environment and/or object distance maps. 
		
		void LightMapInitialization(int render_lightmap_options);	//contorls light map creation
		bool GenerateLightMap(int object_id, boolean delete_lightmap);	//handles light map generation
		void RenderCausticMap(D3DXMATRIX viewMatrix, D3DXMATRIX projMatrix, D3DXVECTOR3 eye, int obj_index);	//renders the photon map
		void RenderShadowMap(D3DXMATRIX viewMatrix, D3DXMATRIX projMatrix, int obj_index);	//renders the shadow map
		void RenderShadowToLightMap(D3DXMATRIX mViewLight, D3DXMATRIX mProjLight);	//renders shadow to light map
		void RenderCausticToLightMap(int object_index);			//renders caustics to light map (photon splatting)
		
		void CreateCausticQuadrilaterals();				//creates the quads used during photon splatting
		
		LPDIRECT3DCUBETEXTURE9 CreateCubeTexture(int size, D3DFORMAT Format);
		LPDIRECT3DTEXTURE9 CreateTexture(int size, D3DFORMAT Format);
		LPDIRECT3DSURFACE9 CreateDepthStencilSurface(int size);

	public:

		Chess(LPDIRECT3DDEVICE9 device);
		void animate(double dt, double t);
		void processMessage(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
		void render();
		HRESULT createManagedResources();
		HRESULT createDefaultResources(wchar_t* effectFileName);
		HRESULT releaseManagedResources();
		HRESULT releaseDefaultResources();

		
		
};
