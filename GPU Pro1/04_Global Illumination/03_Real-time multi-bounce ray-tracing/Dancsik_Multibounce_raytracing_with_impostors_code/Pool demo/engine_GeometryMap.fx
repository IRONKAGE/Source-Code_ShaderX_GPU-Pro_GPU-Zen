#define numberOfIterations 10
#define pi 3.1415926535
#define epsilon 0.0001

float4x4 modelViewProjMatrix;
float4x4 modelMatrix;
float4x4 modelMatrixInverse;
float3 eyePosition;

float4x4 TBN;							//this matrix transforms from the coordinate system of the height map of the water to world space
float4x4 TBNInverse;					//this matrix transforms from world space to the coordinate system of the height map of the water

float4x4 sphereModelMatrix;				//the model matrix of the refractive object
float4x4 sphereModelMatrixInverse;		//the inverse model matrix of the refractive object

float3 referencePointPosition;			//the reference point of the environment distance impostor 

float3 boundingBoxMin, boundingBoxMax;		//bounding box min-max coordinates


float FresnelFactorWater;				// the Fresnel factor of water			
float FresnelFactorRefractor;			// the Fresnel factor of the refractive object	
float FresnelFactorGlass;				// the Fresnel factor of the container	

#define indexOfRefractionWater 0.75
float indexOfRefractionRefractor;


float id;								//object id
float4 kdColor;							//diffuse color

float2 rootNode;						

//wawe parameters
float Time;
float4 SpaceFreq;
float4 TimeFreq;
float4 Amplitudes;
float4 WaveDirX;	
float4 WaveDirZ;
float maximumAmplitude;

float heightMapScale;
float heightMapOffset;

struct Spotlight		//the spotlight structure is used as a point light in this implementation
{
	float3 peakRadiance;		
	float3 position;
	float3 direction;		//unused
	float focus;			//unused
} spotlights[1];
int nSpotlights = 1;


texture diffuseMap, waweNormalMap, waterHeightMap, gaussianFilterTexture, lightMap;

#define SAMPLER2D_LINEAR(samplerMap, txMap);		\
	sampler2D samplerMap = sampler_state {			\
    Texture = <txMap>;								\
    MinFilter = LINEAR;								\
    MagFilter = LINEAR;								\
    MipFilter = LINEAR;								\
    AddressU  = Wrap;								\
    AddressV  = Wrap;								\
};
SAMPLER2D_LINEAR(diffuseMapSampler, diffuseMap);		//diffuse texture of the environment
SAMPLER2D_LINEAR(normalMapSampler, waweNormalMap);			//wawe normals to generate the water height map
SAMPLER2D_LINEAR(waterHeightMapSampler, waterHeightMap);		//water height map
SAMPLER2D_LINEAR(filterTextureSampler, gaussianFilterTexture);		//filter texture to modulate the surroundings of the photon hits in the light map  
SAMPLER2D_LINEAR(lightMapSampler, lightMap);					//light map

texture photonMap;
texture geometryImageTexture, geometryImageNormalMapTexture, minMapTexture, maxMapTexture, linkMapTexture;		

#define SAMPLER2D_POINT(samplerMap, txMap);		\
	sampler2D samplerMap = sampler_state {		\
    Texture = <txMap>;							\
    MinFilter = POINT;							\
    MagFilter = POINT;							\
    MipFilter = POINT;							\
    AddressU  = CLAMP;							\
    AddressV  = CLAMP;							\
};	
SAMPLER2D_POINT(photonMapSampler, photonMap);					//photon map
SAMPLER2D_POINT(geometryImageSampler, geometryImageTexture);					//geometry image representing the refractive object				
SAMPLER2D_POINT(geometryImageNormalMapSampler, geometryImageNormalMapTexture);		//normal vectors for the geometry image
SAMPLER2D_POINT(minMapSampler, minMapTexture);										
SAMPLER2D_POINT(maxMapSampler, maxMapTexture);									//min-max maps for the bounding box hierarchy traversal
SAMPLER2D_POINT(linkMapSampler, linkMapTexture);								//it stores the pointers for the bounding box hierarchy traversal

textureCUBE refractorMapTexture, uvMapTexture, normalMapTexture; 

#define SAMPLERCUBE_LINEAR(samplerMap, txMap);	\
	samplerCUBE samplerMap = sampler_state {	\
    Texture = <txMap>;							\
    MinFilter = LINEAR;							\
    MagFilter = LINEAR;							\
    MipFilter = LINEAR;							\
    AddressU  = BORDER;							\
    AddressV  = BORDER;							\
};

SAMPLERCUBE_LINEAR(refractorMapSampler, refractorMapTexture);				//object distance impostor of the refractive object. It stores surface normals as well.
SAMPLERCUBE_LINEAR(uvMapSampler, uvMapTexture);								//environment distance impostor storing the texture coordinates of the environment
SAMPLERCUBE_LINEAR(glassNormalMapSampler, normalMapTexture);				//object distance impostor of the container










//*********************************************************************************	
//This Shader renders diffuse objects
//*********************************************************************************		
struct TrafoInput
{
    float4 pos			: POSITION;
    float3 normal		: NORMAL;
    float2 tex			: TEXCOORD0;
};

struct TrafoOutput
{
    float4 pos			: POSITION;
    float3 normal		: TEXCOORD2;
    float2 tex			: TEXCOORD0;
    float3 worldPos		: TEXCOORD1;
    float3 view			: TEXCOORD3;
};

TrafoOutput vsTrafo(TrafoInput input)
{
	TrafoOutput output = (TrafoOutput)0;
	output.pos = mul(input.pos, modelViewProjMatrix);

	output.worldPos = mul(input.pos, modelMatrix).xyz;
	
	output.normal = mul(modelMatrixInverse, float4(input.normal.xyz, 0.0));
	output.tex = input.tex;
	output.view = output.worldPos.xyz-eyePosition;
	
	return output;
}

float4 psDiffuseColored(TrafoOutput input) : COLOR0
{
	return kdColor;
}

float3 computeDiffuseColor( float3 worldPos, float3 normal, float2 tex)
{
	float3 lighting = 0.0;
	for(int il=0; il<nSpotlights; il++)
	{
		float3 lightDiff = spotlights[il].position - worldPos.xyz;
		float3 lightDist2 = dot(lightDiff, lightDiff);
		float3 lightDir = normalize(lightDiff);
		/*
		lighting += max(0, dot(lightDir, normal)) * 
					spotlights[il].peakRadiance
					/ (4 * 3.14 * lightDist2);
		*/
	}
	
	//lighting = min(4, lighting);
	
	return saturate( /*lighting **/ kdColor * tex2Dlod(diffuseMapSampler, float4(tex,0,0)) );
}

float4 psDiffuseTextured(TrafoOutput input) : COLOR0
{
	return float4(computeDiffuseColor(input.worldPos, input.normal, input.tex) + tex2Dlod(lightMapSampler, float4(input.tex,0,0)).rgb, 1);
}

technique DiffuseTextured
{
    pass ExamplePass
    {
        VertexShader = compile vs_2_0 vsTrafo();
        PixelShader  = compile ps_3_0 psDiffuseTextured();
    }
}

technique DiffuseColored
{
    pass ExamplePass
    {
        VertexShader = compile vs_2_0 vsTrafo();
        PixelShader  = compile ps_3_0 psDiffuseColored();
    }
}

//*********************************************************************************	
//This Shader stores diffuse objects in an environment distance impostor
//*********************************************************************************	

float4 psCreateEnvironmentMapDiffuse(TrafoOutput input)  : COLOR0
{
	return float4(input.tex, id, length(input.view));	//distance information, texture coordinates and object ids are stored
}

technique CreateEnvironmentMapDiffuse
{
    pass Pass1
    {
        VertexShader = compile vs_2_0 vsTrafo();
        PixelShader  = compile ps_3_0 psCreateEnvironmentMapDiffuse();
    }
}

//*********************************************************************************	
//This Shader stores glass object in an object distance impostor
//*********************************************************************************		

float4 psCreateEnvironmentMapGlass(TrafoOutput input) : COLOR0
{
	return float4(normalize(input.normal), length(input.view));	//distance information and normal vectors are stored
}

technique CreateEnvironmentMapGlass
{
    pass Pass1
    {
		CullMode = CW;
        VertexShader = compile vs_2_0 vsTrafo();
        PixelShader  = compile ps_3_0 psCreateEnvironmentMapGlass();
    }
}

//*********************************************************************************	
//This Shader stores the refractive object in an object distance impostor
//*********************************************************************************	
struct NormalInput
{
    float4 pos			: POSITION;
    float3 normal		: NORMAL;
    float2 tex			: TEXCOORD0;
};

struct NormalOutput
{
    float4 pos			: POSITION;
    float3 normal		: TEXCOORD1;
    float3 modelPos		: TEXCOORD2;
};

NormalOutput vsNormal(NormalInput input)
{
	NormalOutput output = (NormalOutput)0;
	
	output.pos = mul(input.pos, modelViewProjMatrix);
	output.modelPos = input.pos.xyz;				
	output.normal = mul(modelMatrixInverse, float4(input.normal.xyz, 0.0));
	
	return output;
}

float4 psCreateRefractorDistanceMap(NormalOutput input) : COLOR0
{	
	return float4(normalize(input.normal), length(input.modelPos));		//we store normal vectors and distances
}

technique CreateRefractorDistanceMap
{
    pass Pass1
    {
        VertexShader = compile vs_2_0 vsNormal();
        PixelShader  = compile ps_3_0 psCreateRefractorDistanceMap();
    }
}










//*********************************************************************************	
//This Shader creates geometry image (only for star-shaped surfaces)
//*********************************************************************************	
//We take uniformly distributed samples on a unit bounding sphere and we read the object distance impostor in the direction of these samples.
//We assume that the object is star-shaped (convex as seen from its reference point).

float2 halfTexel;

struct psRenderSphereGeometryImageOutput
{
    float4 worldPos	: COLOR0;
    float4 normal   : COLOR1;
};

void vsRenderSphereGeometryImage
(
	in float4 pos : POSITION,
	in float2 tex : TEXCOORD0,
	out float4 posout : POSITION,
	out float2 angles : TEXCOORD0
) 
{
	posout = pos;
	
	//texture coordinates are assigned to angles (transformed from ([0,1],[0,1]) to ([0,2*pi],[0,pi]) ).
	angles = tex * halfTexel * float2(2*pi, pi) + epsilon;		
}

psRenderSphereGeometryImageOutput psRenderSphereGeometryImage
(
	float4 pos : POSITION,
	float2 angles : TEXCOORD0
)
{
	psRenderSphereGeometryImageOutput output;
	
	float sin2PiU, cos2PiU, sinPiV, cosPiV;
	sincos(angles.x, sin2PiU, cos2PiU);
	sincos(angles.y, sinPiV, cosPiV);
	
	//The two angles define a direction...
	float3 lookupDir = float3(cos2PiU*sinPiV, cosPiV, sin2PiU*sinPiV);
	float4 cubeMapData = texCUBE(refractorMapSampler, lookupDir);
	
	
	//...and we read the object distance impostor in that direction
	//to get position normal vector data. 	
	output.normal = float4(normalize(cubeMapData.rgb), 1);	
	output.worldPos = float4(cubeMapData.a * lookupDir, 1);	
	
	return output;
}

technique CreateSphereGeometryImage
{
	pass p0
	{
		VertexShader = compile vs_3_0 vsRenderSphereGeometryImage();
		PixelShader = compile ps_3_0 psRenderSphereGeometryImage();
	}
}

//*********************************************************************************	
//This Shader creates min-max maps
//*********************************************************************************	

struct minMaxData
{
    float4 min	: COLOR0;
    float4 max  : COLOR1;
};

void vsRenderMinMaxMaps
(
	in float4 pos : POSITION,
	in float2 tex : TEXCOORD0,
	out float4 posout : POSITION,
	out float2 texout : TEXCOORD0
) 
{
	posout = pos;
	texout = halfTexel.x*tex + float2(halfTexel.y, halfTexel.y);
}

float2 offset[4];

minMaxData psRenderMinMaxMaps		
(
	float4 pos : POSITION,
	float2 tex : TEXCOORD0
)
{
	minMaxData output;
	
	output.min = float4( tex2Dlod(minMapSampler, float4( tex + offset[0], 0, 0) ).rgb, 1 );
	
	for(int i=1; i<4; i++)
		output.min.xyz = min( output.min.xyz, tex2Dlod(minMapSampler, float4( tex + offset[i], 0, 0) ).rgb );
	
	
	output.max = float4( tex2Dlod(maxMapSampler, float4( tex + offset[0], 0, 0) ).rgb, 1 );
	
	for(int i=1; i<4; i++)
		output.max.xyz = max( output.max.xyz, tex2Dlod(maxMapSampler, float4( tex + offset[i], 0, 0) ).rgb );
	
	return output;
}

technique CreateMinMaxMaps
{
	pass p0
	{
		VertexShader = compile vs_3_0 vsRenderMinMaxMaps();
		PixelShader = compile ps_3_0 psRenderMinMaxMaps();
	}
}

float4x4 transformMatrix;

void vsCopyMinMaxMipMap				//this shader copies the min and max mip maps to a single texture
(
	in float4 pos : POSITION,
	in float2 tex : TEXCOORD0,
	out float4 posout : POSITION,
	out float2 texout : TEXCOORD0
) 
{
	posout = mul(pos, transformMatrix);
	texout = tex + halfTexel;
}

minMaxData psCopyMinMaxMipMap
(
	float4 pos : POSITION,
	float2 tex : TEXCOORD0
)
{	
	minMaxData output;	
	output.min = tex2Dlod(minMapSampler, float4( tex, 0, 0) );
	output.max = tex2Dlod(maxMapSampler, float4( tex, 0, 0) );
	return output;
}

technique CopyMinMaxMipMap
{
	pass p0
	{
		VertexShader = compile vs_3_0 vsCopyMinMaxMipMap();
		PixelShader = compile ps_3_0 psCopyMinMaxMipMap();
	}
}

//*********************************************************************************	
//This Shader creates the link map
//*********************************************************************************	

float4x4 texMatrixHit, texMatrixNode;
float4 linkData;		//x: the height of the geometry image. y: halfTexel for the current level of the link map. z: current level, w: max. level
float halfTexelCurrentLevel;

void vsRenderLinkMap
(
	in float4 pos : POSITION,
	in float2 tex : TEXCOORD0,
	out float4 posout : POSITION,
	out float2 texout : TEXCOORD0,
	out float2 hitTex : TEXCOORD1,
	out float2 globalTex : TEXCOORD2
) 
{
	posout = pos;
	hitTex = mul(pos, texMatrixHit).rg;
	globalTex = mul( pos, texMatrixNode).rg;
	
	float currentHalfTexel = linkData.y;
	texout = tex + currentHalfTexel;						
}

float4 psRenderLinkMap
(
    float4 pos : POSITION,
	float2 tex : TEXCOORD0,
	float2 hitTex : TEXCOORD1,
	float2 globalTex : TEXCOORD2,
	float2 screenCoord : VPOS
) : COLOR
{
	float geomImageSize = linkData.x;
	float currentHalfTexel = linkData.y;
	float currentLevel = linkData.z;
	float maxLevel = linkData.w;
	
	
	//Hit links
	
	float2 hitLink = float2(0,0);
	
	if( currentLevel == maxLevel )
		hitLink = -(screenCoord+1) / geomImageSize;
	else
		hitLink = hitTex + float2(0, 2*halfTexel.y);
	
	if( currentLevel == 0 ) return float4(hitLink, 0, 0);
				
	
	//Miss links
	
	float2 missLink = globalTex;
	
	bool pX = screenCoord.x % 2;	
	bool pY = screenCoord.y % 2;
	
	float2 offset = float2(0, 0);
	
	if( !pX && pY ) offset = float2( 2*halfTexel.x, 0 );
	else if( pX && pY ) offset = float2( 0, -2*halfTexel.y );		
	else if( pX && !pY ) offset = float2( -2*halfTexel.x, 0 );
	else if( !pX && !pY ) missLink = tex2Dlod( geometryImageSampler, float4(tex + currentHalfTexel,0,0) ).ba;

	missLink += offset;
	
	return float4(hitLink, missLink);			
}

technique CreateLinkMap
{
	pass creatingHitLinksAndMissLinks
	{
		VertexShader = compile vs_3_0 vsRenderLinkMap();
		PixelShader = compile ps_3_0 psRenderLinkMap();
	}
}










//*********************************************************************************	
//Functions for ray traversal and intersection computation
//*********************************************************************************	



//*********************************************************************************	
//The Hit function for distance impostors
//*********************************************************************************		
float3 Hit(float3 x, float3 R, samplerCUBE map)
{	
	float rl = texCUBE(map,R).a;	//|r|
	float dp = rl - dot(x, R);
	float3 p = x + R * dp; 
	float ppp = length(p)/texCUBElod(map,float4(p,0)).a;
	float dun =0, dov =0, pun = ppp, pov = ppp;
	
	if (ppp < 1) dun = dp;
	else dov = dp;
	
	float dl = max(dp + rl * (1 - ppp), 0);
	float3 l = x + R * dl;
	
	// iteration
	for(int i = 0; i < numberOfIterations; i++)
	{
		float ddl;
		float llp = length(l)/texCUBElod(map,float4(l,0)).a;
		if (llp < 1)
		{	// undershooting
			dun = dl; pun = llp;
			ddl = (dov == 0) ? rl * (1 - llp) : (dl-dov) * (1-llp)/(llp-pov);
		}
		else
		{	// overshooting
			dov = dl; pov = llp;
			ddl = (dun == 0) ? rl * (1 - llp) : (dl-dun) * (1-llp)/(llp-pun);
		}		
		dl = max(dl + ddl, 0); // avoid flip
		l = x + R * dl;
	}
	return l;
}

//*********************************************************************************	
//Ray-AABB box intersection
//*********************************************************************************	
bool IntersectBoundingBox(float3 rayorigin, float3 raydir, float3 boxmin, float3 boxmax, out float tnear, out float tfar)
{
	// compute intersection of ray with all six bboxplanes
	float3 invR= 1.0 / raydir;
	
	float3 tbot= invR* (boxmin-rayorigin);
	float3 ttop= invR* (boxmax-rayorigin);
	
	// re-order intersections to find smallest and largest on each axis
	float3 tmin= min (ttop, tbot);
	float3 tmax= max (ttop, tbot);
	
	// find the largest tminand the smallest tmax
	float2 t0 = max (tmin.xx, tmin.yz);
	tnear= max (t0.x, t0.y);
	t0 = min (tmax.xx, tmax.yz);
	tfar= min (t0.x, t0.y);
	
	// check for hit
	bool hit;
	if (tnear < tfar && tfar > 0) hit = true;
	else hit = false;
		
	return hit;
	
}

//*********************************************************************************	
//Ray-triangle intersection (Moller-Trumbore method)
//*********************************************************************************	
bool intersectTriangle(float3 edge01, float3 edge02, float3 origin0, float3 raydir, out float2 weight, out float rayparam)
{
	float3 P = cross(raydir, edge02);
	float det = dot(edge01, P);
	if ( det > -0.000001 && det < 0.000001 ) return false;
	float invdet = 1/det;
	
	weight.x =  dot(origin0, P) * invdet;
	if( weight.x < 0 || weight.x > 1 ) return false;
	
	float3 Q = cross(origin0, edge01);
	weight.y = dot(raydir, Q) * invdet;
	if( weight.y < 0 || weight.x + weight.y > 1 ) return false;
	
	rayparam = dot(edge02, Q) * invdet;

	return true;
}

//*********************************************************************************	
//This function reads the environment distance map to obtain the texture coordinates at a given direction
//and queries the diffuse texture and light map of the environment at these coordinates.
//*********************************************************************************	
float4 getEnvironmentColor(float3 direction)
{
	float2 uv = texCUBElod( uvMapSampler, float4(direction,0) ).rg;
	float4 returnData = tex2Dlod( diffuseMapSampler, float4(uv,0,0) );
	returnData.rgb += tex2Dlod( lightMapSampler, float4(uv,0,0) ).rgb;
	return returnData;
}

//*********************************************************************************	
//This function computes the intersection between a ray and an object represented by a geometry image.
//*********************************************************************************	
bool intersectRefractor(float3 rayorigin, float3 raydir, out float4 hitNormal, out float3 hitpoint_refract, out float3 hitpoint_world)
{
	hitNormal = float4(0,0,0,0); hitpoint_refract = float3(0,0,0); hitpoint_world = float3(0,0,0);
	bool returnValue = false;

	//intersection point data. xy: the location of a triangle in the geometry image
	//zw: weight values for the vertices of the triangle to identify the intersection point	
	float4 hitParam = float4(1, 1, 1, -1);
	float rayparam = 10000;						//the "infinite" ray parameter				
	float depth = 0;				
	float2 link = rootNode;			
	float triangleOrientation = -1;	
	
	rayorigin = mul(float4(rayorigin,1), sphereModelMatrixInverse).xyz;
	raydir =  mul(float4(raydir,0), sphereModelMatrixInverse).xyz;
	
	while( depth < 341 )
	{
		float3 boxmin = tex2Dlod(minMapSampler, float4(link,0,0)).rgb;
		float3 boxmax = tex2Dlod(maxMapSampler, float4(link,0,0)).rgb;
		
		float far = 0, near = 0;	
		bool intersectBox = IntersectBoundingBox(rayorigin, raydir,  boxmin, boxmax, near, far);
		
		float4 pointers = tex2Dlod(linkMapSampler, float4(link,0,0)).barg;	//pointers: [miss link, hit link]
		
		if( intersectBox && near<rayparam ) pointers = pointers.barg;		//swap miss link and hit link			
		
		
		if( pointers.r < 0 && pointers.g < 0 )						//leaf								
		{	 
		
			pointers.rg *= -1;
			
			//We select one of the two triangles. The two triangles have a shared edge ( the edge between vertex[0] and vertex[1] ).
			float3 vertex[3];			
			vertex[0] = tex2Dlod(geometryImageSampler, float4(pointers.rg + float2(-halfTexel.x,halfTexel.y),0,0)).rgb;
			vertex[1] = tex2Dlod(geometryImageSampler, float4(pointers.rg + float2(halfTexel.x,-halfTexel.y),0,0)).rgb;
			vertex[2] = tex2Dlod(geometryImageSampler, float4(pointers.rg - halfTexel,0,0)).rgb;
			
			float orient = -1;
			
			float3 edge01 = vertex[1] - vertex[0];
			float3 edge02 = vertex[2] - vertex[0];
			float3 origo0 = rayorigin - vertex[0];
			
			
			float3 n = normalize( cross(edge01, edge02) );			//the normal vector of the triangle plane			
			float d = dot(-origo0, n) / dot(n, raydir);				//intersection with the plane
			
			if( dot( cross( edge01, rayorigin+d*raydir-vertex[0] ) , n ) < 0 )		//if the intersection is in the wrong side	ww inspect the other triangle	
			{ 
				vertex[2] = tex2Dlod(geometryImageSampler, float4(pointers.rg + halfTexel,0,0)).rgb;
				edge02 = vertex[2] - vertex[0];
				orient = 1;
			}
			
			
			float2 weight;
			float param;
			if( intersectTriangle( edge01, edge02, origo0, raydir, weight, param) )		//Moller-Trumbore method		
			{
				if( param > 0 && param < rayparam )
				{
					rayparam = param;
					hitParam = float4(pointers.rg, weight);	
					triangleOrientation = orient;
				}
			}			
										
			pointers = pointers.barg;			//selecting the miss link		
			
		}
		
		if( pointers.r == 0 && pointers.g == 0 ) { depth = 10000; }		
		
		depth += 1;
		link = pointers.rg;
		
	}
	
	if( hitParam.w > -1 )	//if there is an intersection we calculate the normal vector at the found point and the position of the point.
	{
		
		hitNormal.rgb = (1-hitParam.z-hitParam.w) * tex2Dlod(geometryImageNormalMapSampler, float4(hitParam.xy + float2(-halfTexel.x,halfTexel.y),0,0)).rgb
									 + hitParam.z * tex2Dlod(geometryImageNormalMapSampler, float4(hitParam.xy + float2(halfTexel.x,-halfTexel.y),0,0)).rgb 
						  			 + hitParam.w * tex2Dlod(geometryImageNormalMapSampler, float4(hitParam.xy + triangleOrientation*halfTexel,0,0)).rgb;	
		hitpoint_refract = rayorigin + rayparam*raydir;
		hitpoint_world = mul(float4(hitpoint_refract,1), sphereModelMatrix);
		hitNormal.rgb = mul(float4(normalize(hitNormal.rgb),0), sphereModelMatrix);
		returnValue = true;
	}
	
	return returnValue;	
}

//*********************************************************************************	
//This function computes the intersection between a ray and the water surface which is represented by a height map
//*********************************************************************************	
bool intersectWater(float3 rayoriginEnv, float3 rayorigin, float3 raydir, out float3 hitpoint, out float3 hitnormal)
{	
	hitpoint = float3(0,0,0); hitnormal = float3(1,0,0);
	
	float3 intersectionDir = Hit(rayoriginEnv, raydir, glassNormalMapSampler);
	float3 intersectionPoint = intersectionDir * texCUBElod(glassNormalMapSampler, float4(intersectionDir,0)).a / length(intersectionDir) + referencePointPosition;
	intersectionPoint = mul( float4(intersectionPoint, 1), TBNInverse ).xyz;		//intersection point with the container in the height maps space
	
	raydir = normalize( mul( float4(raydir, 0), TBNInverse ).xyz );
	
	float intersectionParam = length( intersectionPoint - rayorigin );
	float2 planeIntersections = float2( -rayorigin.y, 1-rayorigin.y ) / raydir.y;	//intersection with the planes given by equations y=0 and y=1
	
	if( planeIntersections.y < planeIntersections.x ) planeIntersections = planeIntersections.yx;
	if( planeIntersections.y < 0 || intersectionParam < planeIntersections.x ) return false;
	
	
	
	float3 uvmin = rayorigin + planeIntersections.y * raydir;
	float3 uvmax = rayorigin + planeIntersections.x * raydir;
	float Hmin = uvmin.y; float Hmax = uvmax.y;
	float H = 0;
	bool negative = false;
			
	if( planeIntersections.x < 0 )
	{
		negative = true;
		Hmax = rayorigin.y;
	}
	if( intersectionParam < planeIntersections.y ) Hmin = rayorigin.y + intersectionParam * raydir.y;
	
	if( Hmax < Hmin )
	{
		float3 temp = uvmax; uvmax = uvmin; uvmin = temp;
		temp.x = Hmax; Hmax = Hmin; Hmin = temp.x;
	}
	
	float3 q = lerp(uvmin, uvmax, Hmax);

	float2 uv;
	for( int i=0; i<10; i++ )		//binary search
	{			
		H = (Hmin + Hmax)/2;
		uv = lerp(uvmin.xz, uvmax.xz, H);
		
		float h = heightMapScale * tex2Dlod( waterHeightMapSampler, float4(uv,0,0) ).a + heightMapOffset;
				
		if(H>h) Hmax = H;
		else Hmin = H;
	}

	float4 data = tex2D( waterHeightMapSampler, uv );
	
	
	hitpoint = float3(uv.x, H, uv.y);
	 
	if( (negative && length( q - hitpoint ) < 0.1) ||
	    length( intersectionPoint - hitpoint ) < 0.1 
	  ) return false;
	
	
	hitpoint = mul( float4(hitpoint, 1), TBN ).xyz;
	hitnormal = normalize(data.rgb);
	
	return true;
}

bool isAboveWater(float3 pos, out float3 posTBN)
{ 
	posTBN = mul( float4(pos, 1), TBNInverse ).xyz;
	bool returnValue = true;
	if( posTBN.y < heightMapScale*tex2Dlod(waterHeightMapSampler, float4(posTBN.xz,0,0)).a+heightMapOffset ) returnValue = false;
	return returnValue;
}

bool isTotalReflection(float3 ray)
{
	bool returnValue = false;
	if( ray.x==0 && ray.y==0 && ray.z==0 ) returnValue = true;
	return returnValue;
}

//*********************************************************************************	
//This function traces rays through a refractive object. The object is represented by a geometry image.
//It performs double refraction calculation. 
//*********************************************************************************	
void RefractorShading( inout float3 raydir, float3 normal, inout float3 refPos, inout float3 worldPos, float3 refractorPos, out float3 TBNPos, float refrIndex, out bool aboveWater, out float Fresnel )
{
	Fresnel = 1;
	TBNPos = float3(0,0,0);
	aboveWater = false;
	
	float3 refracted_ray = refract(raydir, normal, refrIndex);		//1st refraction
	
	if( isTotalReflection(refracted_ray) )
	{
		raydir = reflect(raydir, normal);
		aboveWater = isAboveWater(worldPos, TBNPos);
	}
	else												//2nd refraction
	{
		Fresnel -= FresnelFactorRefractor + ( 1 - FresnelFactorRefractor ) * pow( 1 - dot( normal, -raydir ), 5);
				
		float4 hitNormal;
		intersectRefractor(worldPos-0.3*normal, refracted_ray, hitNormal, refractorPos, worldPos);
		
		refPos = worldPos - referencePointPosition;
		
		aboveWater = isAboveWater(worldPos, TBNPos);
		
		if( aboveWater ) refrIndex = 1 / indexOfRefractionRefractor;
		else refrIndex = indexOfRefractionWater / indexOfRefractionRefractor; 
		
		normal = -hitNormal.rgb;
		raydir = refract(refracted_ray, normal, refrIndex);
		
		if( isTotalReflection(raydir) )
		{
			raydir = reflect(refracted_ray, normal);
		}
		else
			Fresnel *= ( 1 - FresnelFactorRefractor + ( 1 - FresnelFactorRefractor ) * pow( 1 - dot( normal, -refracted_ray ), 5) );
	}		
}





















struct WaterInput
{
    float4 pos			: POSITION;
    float2 tex			: TEXCOORD0;
    float4 normal		: NORMAL;
};

struct WaterOutput
{
    float4 pos			: POSITION;
    float3 worldPos		: TEXCOORD0;
    float3 view			: TEXCOORD1;
    float3 refPos		: TEXCOORD2;
    float3 normal		: TEXCOORD3;
};

//*********************************************************************************	
//Water surface shader
//*********************************************************************************	

WaterOutput vsWater(WaterInput input)
{	
	WaterOutput output;
	
	float4 heightMapData = tex2Dlod(waterHeightMapSampler, float4( input.tex, 0, 0));
	
	float4 modelPos = input.pos;
	modelPos.y -= maximumAmplitude;
	modelPos.y += heightMapData.a;
	
	output.pos = mul(modelPos, modelViewProjMatrix);
	output.worldPos = mul(modelPos, modelMatrix).xyz;
	output.view = output.worldPos - eyePosition;
	output.refPos = output.worldPos - referencePointPosition;
	output.normal = normalize(heightMapData.rgb);

	return output;
}



float4 psWater(WaterOutput input) : COLOR0
{
	float3 view = normalize(input.view);
	float3 normal = normalize(input.normal);
	float3 refPos = input.refPos;
	float3 refracted_ray = refract(view, normal, indexOfRefractionWater);	
	float3 reflected_ray = reflect(view, normal);
	float4 reflectedColor = float4(0,0,0,0);
	float4 refractedColor = float4(0,0,0,0);
	
	
	//reflection direction
	
	float4 hitData;
	float3 hitRefractor, hitWorld, TBNPos;
	bool aboveWater;
	float Fresnel = 1;
	if( intersectRefractor(input.worldPos, reflected_ray, hitData, hitRefractor, hitWorld) )	//there is an intersection with the refractive object
	{		//double refraction computation
			refPos = hitWorld - referencePointPosition;
			reflectedColor = float4(0,0.05,0.17,0);
			RefractorShading( reflected_ray, hitData.rgb, refPos, hitWorld, hitRefractor, TBNPos, indexOfRefractionRefractor, aboveWater, Fresnel );
	}
	
	reflectedColor += Fresnel * getEnvironmentColor( Hit(refPos, reflected_ray, uvMapSampler) );
	
	
	//reraction direction
	
	Fresnel = 1;
	refPos = input.refPos;
	if( intersectRefractor(input.worldPos, refracted_ray, hitData, hitRefractor, hitWorld) )		//there is an intersection with the refractive object
	{		//double refraction computation
			refPos = hitWorld - referencePointPosition;
			refractedColor = float4(0,0.05,0.17,0);
			RefractorShading( refracted_ray, hitData.rgb, refPos, hitWorld, hitRefractor, TBNPos, indexOfRefractionRefractor/indexOfRefractionWater, aboveWater, Fresnel );
	}
	 
	//At the reflection direction we neglect the intersection computation with the container.
	 
	float3 intersectionDir = Hit(refPos, refracted_ray, glassNormalMapSampler);
	hitData = texCUBElod( glassNormalMapSampler, float4(intersectionDir,0));
	refPos = intersectionDir * hitData.a / length(intersectionDir);
	hitData.rgb = normalize(-hitData.rgb);
		
	float3 temp_ray = refract(refracted_ray, hitData.rgb, 1/indexOfRefractionWater);
			
	if( isTotalReflection(temp_ray) )
		temp_ray = reflect(refracted_ray, hitData.rgb);
	else
		Fresnel *= 1 - FresnelFactorWater + ( 1 - FresnelFactorWater ) * pow( 1 - dot( hitData.rgb, -refracted_ray ), 5);
	
	refractedColor += Fresnel * getEnvironmentColor( Hit(refPos, temp_ray, uvMapSampler) );
	
	float F = FresnelFactorWater + ( 1 - FresnelFactorWater ) * pow( 1 - dot( normal, -view ), 5 );
	return lerp(refractedColor, reflectedColor, F);
}

technique Water
{
    pass ExamplePass
    {
        VertexShader = compile vs_3_0 vsWater();
        PixelShader  = compile ps_3_0 psWater();
    }
}

//*********************************************************************************	
//Shader of the water volume
//*********************************************************************************	

WaterOutput vsUnderWater(WaterInput input)
{	
	WaterOutput output;
	
	float4 heightMapData = tex2Dlod(waterHeightMapSampler, float4( input.tex, 0, 0));
	
	float4 modelPos = input.pos;
	
	//the upper vertices are part of the water surface therefore we have to modify their positions.
	if( modelPos.y > -0.01 )	 
	{
		modelPos.y -= maximumAmplitude;
		modelPos.y += heightMapData.a;
	}
	
	output.pos = mul(modelPos, modelViewProjMatrix);
	output.worldPos = mul(modelPos, modelMatrix).xyz;
	output.view = output.worldPos - eyePosition;
	output.refPos = output.worldPos - referencePointPosition;
	output.normal = mul( modelMatrixInverse, float4(input.normal.xyz, 0.0) );
	
	return output;
}

float4 psUnderWater(WaterOutput input) : COLOR0
{	
	float3 view = normalize(input.view);
	float3 normal = normalize(input.normal);
	float3 refPos = input.refPos;
	float Fresnel_refract = 1;
	
	float3 refracted_ray = refract(view, normal, indexOfRefractionWater);
	float3 reflected_ray = reflect(view, normal);
	
	float4 refractedColor = float4(0,0,0,0);
	float4 reflectedColor = getEnvironmentColor( Hit(refPos, reflected_ray, uvMapSampler) );
	
	
	//we need to follow the refracted ray through the water volume
	
	float4 hitData;
	float3 hitRefractor, hitWorld, hitTBN;
	bool isIntersectionWithRefractor = intersectRefractor(input.worldPos, refracted_ray, hitData, hitRefractor, hitWorld);
	bool aboveWater = isAboveWater(hitWorld, hitTBN);
	
	
	
	if( isIntersectionWithRefractor )	//if the ray intersects the refractive object... 
	{		
			aboveWater = isAboveWater(hitWorld, hitTBN);
			if( !aboveWater )	//...and the refractive object is underwater, the computation of double refraction is necessary
			{
				refractedColor = float4(0,0.05,0.17,0);
				refPos = hitWorld - referencePointPosition;
				RefractorShading( refracted_ray, hitData.rgb, refPos, hitWorld, hitRefractor, hitTBN, indexOfRefractionRefractor/indexOfRefractionWater, aboveWater, Fresnel_refract );
			}
			else
				isIntersectionWithRefractor = false;
	}
	if( !isIntersectionWithRefractor ) hitTBN = mul( float4(input.worldPos, 1), TBNInverse ).xyz;
	
	
	//intersection computation with the water surface
	bool isIntersectionWithWater = intersectWater(refPos, hitTBN, refracted_ray, hitWorld, hitData.rgb);
	float3 waterNormal = -hitData.rgb;
	float3 waterHitPoint = hitWorld;
	float3 refPosWater = hitWorld - referencePointPosition;
	
	
	float F = 1;
	float4 refl_col = float4(0,0,0,0);
	
	
	//if the ray intersects the water surface we need to continue the search both in the reflection and refraction directions
	float3 newDir = refracted_ray;
	if( isIntersectionWithWater )		//inspecting the reflection direction		
	{	
		newDir = reflect(refracted_ray, waterNormal);
		refPos = refPosWater;
		
		if( intersectRefractor(waterHitPoint, newDir, hitData, hitRefractor, hitWorld) )	//in case of an intersection with the refractive object->double refraction
		{	
			refl_col = float4(0,0.05,0.17,0);
			refPos = hitWorld - referencePointPosition;
			RefractorShading( newDir, hitData.rgb, refPos, hitWorld, hitRefractor, hitTBN, indexOfRefractionRefractor/indexOfRefractionWater, aboveWater, F );
		}
		
	}
	
	//computing the intersection between the reflected ray and the glass container
	float3 intersectionDir = Hit(refPos, newDir, glassNormalMapSampler);
	hitData = texCUBElod( glassNormalMapSampler, float4(intersectionDir,0));
	refPos = intersectionDir * hitData.a / length(intersectionDir);
	hitData.rgb = -normalize(hitData.rgb);
		
	//the reflected ray leaves the weater volume	
	float3 temp_ray = refract(newDir, hitData.rgb, 1/indexOfRefractionWater);
			
	if( isTotalReflection(temp_ray) )
		temp_ray = reflect(newDir, hitData.rgb);
	else
		F *= 1 - FresnelFactorWater + ( 1 - FresnelFactorWater ) * pow( 1 - dot( hitData.rgb, -temp_ray ), 5);
	
	//finally we search in the environment impostor to obtain the incoming radiance from the reflection direction
	refl_col += F * getEnvironmentColor( Hit(refPos, newDir, uvMapSampler) );	
		
	
	if( isIntersectionWithWater )	//if the ray intersects the water surface...
	{
		newDir = refract(refracted_ray, waterNormal, 1/indexOfRefractionWater);		//...we need to follow the refracted direction as well
		
		if( !isTotalReflection(newDir) )
		{
			refPos = refPosWater;
			float4 refr_col = float4(0,0,0,0);
			
			if( intersectRefractor(waterHitPoint, newDir, hitData, hitRefractor, hitWorld) )	//intersection with the refractive object
			{	
				refr_col = float4(0,0.05,0.17,0);
				refPos = hitWorld - referencePointPosition;
				RefractorShading( newDir, hitData.rgb, refPos, hitWorld, hitRefractor, hitTBN, indexOfRefractionRefractor/indexOfRefractionWater, aboveWater, F );	//double refraction
			}
			
			//we neglect the intersection with the translucent container
			
			F = FresnelFactorWater + ( 1 - FresnelFactorWater ) * pow( 1 - dot( waterNormal, -refracted_ray ), 5 );
			refr_col += getEnvironmentColor( Hit(refPos, newDir, uvMapSampler) );
			refractedColor += Fresnel_refract*lerp(refr_col, refl_col, F);
		}
		else  refractedColor += Fresnel_refract*refl_col;
	}
	else  refractedColor += Fresnel_refract*refl_col;
	
	
	F = FresnelFactorWater + ( 1 - FresnelFactorWater ) * pow( 1 - dot( normal, -view ), 5 );
	
	return lerp(refractedColor, reflectedColor, F);
}


technique UnderWater
{
    pass ExamplePass
    {
        VertexShader = compile vs_3_0 vsUnderWater();
        PixelShader  = compile ps_3_0 psUnderWater();
    }
}



















//*********************************************************************************	
//The refractive object's shader
//*********************************************************************************		

struct RefractorInput
{
	float4  pos			: POSITION;
	float3 normal		: NORMAL;
};

struct RefractorOutput
{
	float4 pos			: POSITION;
	float3 worldPos		: TEXCOORD0;
	float3 normal		: TEXCOORD1;
	float3 view			: TEXCOORD2;
	float3 refPos		: TEXCOORD3;
	float3 refractorPos : TEXCOORD4;
	
};

RefractorOutput vsRefractor(RefractorInput input)
{
	RefractorOutput output;

	output.pos = mul( input.pos, modelViewProjMatrix );
	output.worldPos = mul( input.pos, modelMatrix ).xyz;
	output.normal =	mul( modelMatrixInverse, float4(input.normal.xyz, 0.0) );
	output.view = output.worldPos - eyePosition;
	output.refPos = output.worldPos - referencePointPosition;
	output.refractorPos = input.pos.xyz;
	
	return output;
}

float4 psRefractor(RefractorOutput input) : COLOR0
{	
	float3 raydir = normalize(input.view);
	float3 refPos = input.refPos;
	float3 worldPos = input.worldPos;
	float3 TBNPos;
	bool aboveWater = true;
	float refrIndex = indexOfRefractionRefractor;
	float Fresnel = 1;
	float F = 0;
	
	float4 refractedColor = float4(0,0,0,1);
	float4 reflectedColor = float4(0,0,0,1);
	float3 hitNormal, rrr;
	
	float3 temp_ray = raydir;
	
	//we follow the ray through the refractive object (double refraction computation)
	RefractorShading( raydir, normalize(input.normal), refPos, worldPos, input.refractorPos, TBNPos, refrIndex, aboveWater, Fresnel );

	float3 refHit; float3 refNormal;
	if( intersectWater(refPos, TBNPos, raydir, refHit, refNormal) )	//the ray intersects the water surface
	{	
		refPos = refHit - referencePointPosition;
		
		if( !aboveWater )
		{
			refNormal = -refNormal;
			refrIndex = 1 / indexOfRefractionWater;
		}
		else
			refrIndex = indexOfRefractionWater;
		
		F = FresnelFactorWater + ( 1 - FresnelFactorWater ) * pow( 1 - dot( refNormal, -raydir ), 5);
		
		reflectedColor = getEnvironmentColor( Hit(refPos, reflect(raydir, refNormal), uvMapSampler) );
		
		temp_ray = refract(raydir, refNormal, refrIndex);
		
		if( isTotalReflection(temp_ray) ) refractedColor = reflectedColor;
		else aboveWater = !aboveWater;
	}
	
	//ray - container intersection
	if( !aboveWater && !isTotalReflection(temp_ray) )
	{
		float3 intersectionDir = Hit(refPos, temp_ray, glassNormalMapSampler);
		float4 intersectionData = texCUBE( glassNormalMapSampler, intersectionDir);
		refPos = intersectionDir * intersectionData.a / length(intersectionDir);
		intersectionData.rgb = -intersectionData.rgb;
			
		raydir = refract(temp_ray, intersectionData.rgb, 1/indexOfRefractionWater);
			
		if( isTotalReflection(raydir) ) raydir = reflect(temp_ray, intersectionData.rgb);
	}
	
	
	refractedColor = getEnvironmentColor( Hit(refPos, raydir, uvMapSampler) );
	
	return Fresnel * lerp(refractedColor, reflectedColor, F) + float4(0,0.05,0.17,0);
}

technique Refractor
{
    pass ExamplePass
    {
        VertexShader = compile vs_3_0 vsRefractor();
        PixelShader  = compile ps_3_0 psRefractor();
    }
}












//*********************************************************************************	
//The translucent glass container's shader
//*********************************************************************************		

struct GlassInput
{
	float4  pos			: POSITION;
	float3 normal		: NORMAL;
};

struct GlassOutput
{
	float4 pos			: POSITION;
	float3 worldPos		: TEXCOORD0;
	float3 normal		: TEXCOORD1;
	float3 view			: TEXCOORD2;
	float3 refPos		: TEXCOORD3;	
};

GlassOutput vsGlass(GlassInput input)
{
	GlassOutput output;

	output.pos = mul( input.pos, modelViewProjMatrix );
	output.worldPos = mul( input.pos, modelMatrix ).xyz;
	output.normal =	mul( modelMatrixInverse, float4(input.normal.xyz, 0.0) );
	output.view = output.worldPos - eyePosition;
	output.refPos = output.worldPos - referencePointPosition;
	
	return output;
}

float4 psGlass(GlassOutput input) : COLOR0
{
	float3 raydir = normalize(input.view);
	float3 normal = normalize(input.normal);
	
	float dotProduct = dot( normal, -raydir );
	
	if( dotProduct < 0 )
	{
		normal = -normal;
		dotProduct = -dotProduct;
	}
	
	//the container is not refractive
	//therefore it is enough to calculate the radiance incoming from the reflection direction.
	//The translucence is handled by blending.
	float3 hitDirection = Hit(input.refPos, reflect(raydir, normal), uvMapSampler);
	float3 reflColor = getEnvironmentColor( hitDirection ).rgb;			
	
	float F = FresnelFactorGlass + ( 1 - FresnelFactorGlass ) * pow( 1 - dotProduct, 5);
	
	return float4(reflColor, F); 
		
}

technique Glass
{
    pass Pass0
    {
		AlphaBlendEnable = true;
		SrcBlend = SRCALPHA;
		DestBlend = INVSRCALPHA;
		CullMode = NONE;
        VertexShader = compile vs_3_0 vsGlass();
        PixelShader  = compile ps_3_0 psGlass();
    }
}



















//*********************************************************************************	
//This shader creates the height map of the water surface
//*********************************************************************************	

struct vsWaterHeightMapInput
{
	float4 pos			: POSITION;
	float2 tex			: TEXCOORD0;
};

struct vsWaterHeightMapOutput
{
	float4 pos			: POSITION;
	float2 modelPos		: TEXCOORD0;
};

vsWaterHeightMapOutput vsWaterHeightMap(vsWaterHeightMapInput input)
{
	vsWaterHeightMapOutput OUT;
	
	float2 minus = boundingBoxMax.xz - boundingBoxMin.xz;
	
	OUT.pos = input.pos;
	OUT.modelPos = minus*input.tex + boundingBoxMin.xz;
	
	return OUT;
}

float4 psWaterHeightMap(vsWaterHeightMapOutput input) : COLOR0
{	
	float4 Phase = (WaveDirX * input.modelPos.x + WaveDirZ * input.modelPos.y) * SpaceFreq + Time * TimeFreq;
	    
	float4 Cos,Sin;
	sincos(Phase, Sin, Cos);
	float WaveHeight = dot(Sin, Amplitudes);
	
	float2 normalTex = input.modelPos.xy * 0.02;

	float4 CosWaveHeight = Cos * Amplitudes * SpaceFreq;		
	
	float3 tangent  = normalize(float3(1, dot(CosWaveHeight,WaveDirX), 0));	
	float3 binormal = normalize(float3(0, dot(CosWaveHeight,WaveDirZ), 1));
	float3 normal =  normalize(cross(binormal, tangent)); 
	
	
	float3x3 TBNMatrix;
	TBNMatrix[0] = tangent;
	TBNMatrix[1] = normal;
	TBNMatrix[2] = binormal;
	
	half4 t0 = tex2Dlod(normalMapSampler, float4(normalTex + float2(-0.07, 0) * Time,0,0));
	half4 t1 = tex2Dlod(normalMapSampler, float4(normalTex + float2(0.05, 0.0285) * Time,0,0));
	half4 t2 = tex2Dlod(normalMapSampler, float4(normalTex + float2(0.04, -0.235) * Time,0,0));
	half4 t3 = tex2Dlod(normalMapSampler, float4(normalTex + float2(0.026, 0.026) * Time,0,0));
	
	float3 norm = ( (t0+t1+t2+t3) * 0.5 - 1.0) + float3(0, 4, 0);
	norm.b *= 0.071;		//the b component is too big
	normal = mul(norm, TBNMatrix); 
	normal = normalize(normal);
	
	return float4(norm, WaveHeight + maximumAmplitude);
	
};

technique CreateWaterHeightMap
{
	pass P0
	{
		VertexShader = compile vs_3_0 vsWaterHeightMap();
		PixelShader = compile ps_3_0 psWaterHeightMap();
	}	
};










float getPower(float3 worldPos)
{
	float3 toLight = spotlights[0].position - worldPos;
	float lightDist2 = dot(toLight, toLight);
	float3 rad = spotlights[0].peakRadiance / (4 * pi * lightDist2);
	return (rad.x + rad.y + rad.z) * 0.33;			//avarage
}

//*********************************************************************************	
//This shader renders the water surface to the photon map.
//*********************************************************************************	
float4 psPhotonMapWater(WaterOutput input) : COLOR0
{
	float4 returnColor = float4(0,0,0,0);
	
	float3 raydir = normalize(input.view);
	float3 normal = normalize(input.normal);
	
	float F = FresnelFactorWater + ( 1 - FresnelFactorWater ) * pow( 1 - dot( normal, -raydir ), 5 );
	
	float power = getPower(input.worldPos);
	
	float index = indexOfRefractionWater;
	
	bool aboveWater = false;
	
	//Depending on the fresnel term we choose which direction we need to follow
	if( F < 0.5 )	//the refraction direction is more significant
	{
		raydir = refract(raydir, normal, index);
		power *= (1-F);
		index = indexOfRefractionRefractor / indexOfRefractionWater;
	}
	else		//the reflection direction is more significant
	{
		raydir = reflect(raydir, normal);
		power *= F;
		index = indexOfRefractionRefractor;
		aboveWater = true;
	}
	
	
	
	float4 hitData;
	float3 hitRefractor, hitWorld;
	float3 TBNPos;
	float3 refPos = input.refPos;
	
	if( intersectRefractor(input.worldPos, raydir, hitData, hitRefractor, hitWorld) )	//intersection with the refractive object
	{
			refPos = hitWorld - referencePointPosition;
			RefractorShading( raydir, normalize(hitData.rgb), refPos, hitWorld, hitRefractor, TBNPos, index, aboveWater, F );
			power *= F;
	}
	
	//intersection with the container
	float3 intersectionDir = Hit(refPos, raydir, glassNormalMapSampler);
	hitData = texCUBElod( glassNormalMapSampler, float4(intersectionDir,0));
	refPos = intersectionDir * hitData.a / length(intersectionDir);
	normal = -normalize(hitData.rgb);

	
	if( !aboveWater )	//if the ray leaves the water volume refraction calculation is needed
	{
		power *= 1-FresnelFactorWater + ( 1 - FresnelFactorWater ) * pow( 1 - dot( normal, -raydir ), 5 );
		index = 1 / indexOfRefractionWater;
		raydir = refract(raydir, normal, index);
			
		if ( isTotalReflection(raydir) ) power = 0;		//we neglect the total reflection
	}
	
	power *= 1-FresnelFactorGlass + ( 1 - FresnelFactorGlass ) * pow( 1 - dot( normal, -raydir ), 5 );
	
	//we save the location of the photon hit in texture space 	
	returnColor = texCUBElod( uvMapSampler, float4(Hit(refPos, raydir, uvMapSampler),0) );
	
	returnColor.a = max(power, 0);	//the power of the photon is saved as well
	return returnColor;		
}
 
technique PhotonMapWater
{
	pass P0
	{
		VertexShader = compile vs_3_0 vsWater();
		PixelShader = compile ps_3_0 psPhotonMapWater();
	}	
};

//*********************************************************************************	
//This shader renders the refractive object to the photon map
//*********************************************************************************	
float4 psPhotonMapRefractor(RefractorOutput input) : COLOR0
{
	float4 returnColor = float4(0,0,0,0);
	
	float3 raydir = normalize(input.view);
	float3 normal = normalize(input.normal);
	
	float F = FresnelFactorRefractor + ( 1 - FresnelFactorRefractor ) * pow( 1 - dot( normal, -raydir ), 5 );
	float power = (1-F) * getPower(input.worldPos);
	float3 refPos = input.refPos;
	float refrIndex =  indexOfRefractionRefractor;
	
	float3 hitWorld = input.worldPos;
	float3 TBNPos;
	bool aboveWater = true;
	F = 1;
	
	//ray traversal through the refractive object
	RefractorShading( raydir, normal, refPos, hitWorld, input.refractorPos, TBNPos, refrIndex, aboveWater, F );
	power *= F;

	if( intersectWater(refPos, TBNPos, raydir, hitWorld, normal) )	//intersection with the water surface
	{	
		refPos = hitWorld - referencePointPosition;
		
		if( !aboveWater )
		{
			normal = -normal;
			refrIndex = 1 / indexOfRefractionWater;
		}
		else
			refrIndex = indexOfRefractionWater;
		
		F = FresnelFactorWater + ( 1 - FresnelFactorWater ) * pow( 1 - dot( normal, -raydir ), 5);
		
		//Depending on the fresnel term we choose which direction we need to follow
		
		float3 temp_ray;
		if( F < 0.5 )	
		{
			refrIndex = 1 / indexOfRefractionWater;
			temp_ray = refract(raydir, normal, refrIndex);
			F = 1-F;
			aboveWater = !aboveWater;
		}
		else
			temp_ray = reflect(raydir, normal);
		
		
		if ( isTotalReflection(temp_ray) )
		{
			temp_ray = reflect(raydir, normal);
			F = 1;
			aboveWater = !aboveWater;
		}
		
		power *= F;
		raydir = temp_ray;
	}
	
	//intersection with the container
	float3 intersectionDir = Hit(refPos, raydir, glassNormalMapSampler);
	float4 hitData = texCUBElod( glassNormalMapSampler, float4(intersectionDir,0));
	refPos = intersectionDir * hitData.a / length(intersectionDir);
	normal = -normalize(hitData.rgb);
	
	if( !aboveWater )	//if the ray leaves the water volume refraction calculation is needed
	{
		power *= 1-FresnelFactorWater + ( 1 - FresnelFactorWater ) * pow( 1 - dot( normal, -raydir ), 5 );
		refrIndex = 1 / indexOfRefractionWater;
		raydir = refract(raydir, normal, refrIndex);
	
		if ( isTotalReflection(raydir) ) power = 0;		//we neglect total reflection
	}
	
	power *= 1-FresnelFactorGlass + ( 1 - FresnelFactorGlass ) * pow( 1 - dot( normal, -raydir ), 5 );
	
	//the location of the photon hit in texture space 	
	returnColor = texCUBE(uvMapSampler, Hit(refPos, raydir, uvMapSampler) );
	
	returnColor.a = max(0, power);		//power of the photon
	return returnColor;	
}

technique PhotonMapRefractor
{
	pass P0
	{
		VertexShader = compile vs_3_0 vsRefractor();
		PixelShader = compile ps_3_0 psPhotonMapRefractor();
	}	
};

//*********************************************************************************	
//This shader renders caustics to the light map
//*********************************************************************************	
struct causticInput
{
    float4 pos			: POSITION;
    float4 tex			: TEXCOORD0;
};

struct causticOutput
{
    float4 pos			: POSITION;
    float2 tex			: TEXCOORD0;
    float power			: TEXCOORD1;
};

#define HALF 1/512

causticOutput vsRenderPhotonHit(causticInput input)
{
	causticOutput output;

	float2 size = input.tex.xy;
	float2 ph_uv = input.pos.xy;
	output.tex = input.tex.zw;
	
	float4 ph = tex2Dlod(photonMapSampler, float4( ph_uv, 0, 0 ));

    output.pos = float4( 2 * ph.r - 1 + size.x + HALF,
						 -2 * ph.g + 1 + size.y - HALF,
						 0,
						 1 );

    if( ph.b < 1 ) output.pos.z = -100;
	
	output.power = ph.a;
	
	return output;
}

#define causticIntensity 0.01

float4 psRenderPhotonHit(causticOutput input) : COLOR
{	
	float4 retColor;
	
	float4 weight = tex2Dlod( filterTextureSampler, float4(input.tex,0,0) );
	
	retColor = weight * input.power * causticIntensity;	
	
	return retColor;
}

technique renderPhotonHit
{
	pass P0
	{
		AlphaBlendEnable = true;
		SrcBlend = One;
		DestBlend = One;
		ZWriteEnable = false;
		VertexShader = compile vs_3_0 vsRenderPhotonHit();
		PixelShader = compile ps_3_0 psRenderPhotonHit();
	}	
};