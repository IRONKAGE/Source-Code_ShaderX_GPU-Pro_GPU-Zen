#define NumberOfIterations 8
#define HALF	0.5f  / 512.0f

float FresnelFactor;			//fresnel factor of the glass pieces


float IndexOfRefraction;		//IOR of the glass pieces

float id;					//object index
float id_moving;			//index of the moving piece

float3 selected_obj_position;	//position of the rendered object
float3 moving_obj_position;		//position of the moving chess piece

float3 moving_obj_boundingMin, moving_obj_boundingMax;		//min. and max coordinates of the bounding box of the moving piece
float3 selected_obj_boundingMin, selected_obj_boundingMax;	//min. and max coordinates of the bounding box of the rendered piece

float4x4 modelViewProjMatrix;		//transforms from model space to projection space
float4x4 modelMatrix;				//transforms from model space to world space
float4x4 modelMatrixInverse;		//inverse model transformation
float4x4 viewProjInverseMatrix;		//transforms from view space to projection space	
float4x4 LightModelViewProjTexBias;	//transforms from the light sources model space to projection space

float4 chesstable_data;				//chessboard boundingbox min.xz, max.xz

float3 EyePos;						//camera eye position
float3 LightDir;					//light direction
float Power, CausticsIntensity;		//power of the light source and intensity of caustics




/****************************************************************/
/*Texture maps & samplers*/
/****************************************************************/
texture FilterTexture;							//light map of the static environment (or the height map impostor of the moving piece)
texture FilterTexture_moving;					//light map of the dynamic environment 
texture diffuseMap;								//diffuse texture of the chessboard (or the height map impostor of the rendered object)
texture causticMapTexture;						//photon map
texture PowerOfSnippetMaptexture;				//gaussian filter texture

#define SAMPLER2D_LINEAR(samplerMap, txMap);		\
	sampler2D samplerMap = sampler_state {			\
    Texture = <txMap>;								\
    MinFilter = LINEAR;								\
    MagFilter = LINEAR;								\
    MipFilter = LINEAR;								\
    AddressU  = BORDER;								\
    AddressV  = BORDER;								\
};
SAMPLER2D_LINEAR(diffuseMapSampler, diffuseMap);
SAMPLER2D_LINEAR(diffuseMapFilter, FilterTexture);
SAMPLER2D_LINEAR(diffuseMapFilter_moving, FilterTexture_moving);
SAMPLER2D_LINEAR(PowerOfSnippetMap, PowerOfSnippetMaptexture);


#define SAMPLER_POINT(samplerMap, txMap);		\
	sampler2D samplerMap = sampler_state {		\
    Texture = <txMap>;							\
    MinFilter = POINT;							\
    MagFilter = POINT;							\
    MipFilter = NONE;							\
    AddressU  = BORDER;							\
    AddressV  = BORDER;							\
};	
SAMPLER_POINT(causticmap, causticMapTexture);

/****************************************************************/
/*Cube Maps
/****************************************************************/

textureCUBE refractorCubeTexture, refractorCubeTexture_moving_obj;		//object distance impostors of the rendered and the moving pieces
textureCUBE environmentCubeTexture, environmentCubeTexture_moving_obj;	//environment distance impostors of the rendered and the moving pieces

#define SAMPLER_LINEAR(samplerMap, txMap);		\
	samplerCUBE samplerMap = sampler_state {	\
    Texture = <txMap>;							\
    MinFilter = LINEAR;							\
    MagFilter = LINEAR;							\
    MipFilter = LINEAR;							\
    AddressU  = BORDER;							\
    AddressV  = BORDER;							\
};

SAMPLER_LINEAR(refrmap,refractorCubeTexture);
SAMPLER_LINEAR(refrmap_moving,refractorCubeTexture_moving_obj);

SAMPLER_LINEAR(envmap,environmentCubeTexture);
SAMPLER_LINEAR(envmap_moving,environmentCubeTexture_moving_obj);
 
/****************************************************************/
/*Sky Map*/
/****************************************************************/

textureCUBE skyCubeTexture;			//sky texture

SAMPLER_LINEAR(skymap,skyCubeTexture);

texture shadowMapTexture;			//shadow map

/****************************************************************/
/*Shadow Map*/
/****************************************************************/
//SAMPLER_POINT(shadowMap, shadowMapTexture);
sampler2D shadowMap = sampler_state
{
    Texture = <shadowMapTexture>;
    MinFilter = Linear;
    MagFilter = Linear;
    MipFilter = Linear;
    AddressU  = BORDER;
    AddressV  = BORDER;
    BorderColor = 0xffffffff;
};








/****************************************************************/
//This shader creates an object distance impostor.
//It stores surface normals and distances from the object's pivot point.
//The result is stored in a cube map.
/****************************************************************/

void vsComputeRefractor
(
	in float4 pos : POSITION,
	in float3 normal : NORMAL,
	out float4 posout : POSITION,
	out float3 normalout : TEXCOORD0,
	out float3 modelposout : TEXCOORD1
)
{	
	posout = mul( pos, modelViewProjMatrix );
	modelposout = pos.xyz;
	normalout = mul( modelMatrixInverse, float4(normal.xyz, 0.0) );
}

float4 psComputeRefractor
(
	float4 pos : POSITION,
	float3 normal : TEXCOORD0,
	float3 modelpos : TEXCOORD1
) : COLOR
{
	return float4(normalize(normal), length(modelpos));
}




/****************************************************************/
//This shader creates a height map impostor.
//It stores surface normals and distances from the object's plane of symmetry.
//In this implementation the plane of symmetry has the equation 'x=0'.
//The result is stored in a 2D map.
/****************************************************************/
float4 psComputeHeightMap
(
	float4 pos : POSITION,
	float3 normal : TEXCOORD0,
	float4 modelpos : TEXCOORD1
) : COLOR
{	
	return float4( normalize(normal), modelpos.x );	
}




/****************************************************************/
//This function determines the color of the chess piece.
/****************************************************************/

float3 GetColor(float obj_index) {
	return 0.00175 * float3(obj_index, 33-obj_index, 0);
}




/****************************************************************/
//This function computes intersection between a ray and a distance impostor.
/****************************************************************/

float3 Hit(float3 x, float3 R, samplerCUBE map) {
	
	float rl = texCUBElod(map,float4(R,0)).a;	//|r|
	float dp = rl - dot(x, R);
	float3 p = x + R * dp;
	float ppp = length(p)/texCUBElod(map,float4(p,0)).a;
	float dun =0, dov =0, pun = ppp, pov = ppp;
	
	if (ppp < 1) dun = dp;
	else dov = dp;
	
	float dl = max(dp + rl * (1 - ppp), 0);
	float3 l = x + R * dl;
	
	
	// iteration
	for(int i = 0; i < NumberOfIterations; i++)
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




/****************************************************************/
//Ray - AABB intersection computation.
/****************************************************************/

bool IntersectBoundingBox(float3 rayorigin,float3 raydir,float3 boxmin,float3 boxmax,out float tnear,out float tfar)
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




//****************************************************************/
//Ray - height map intersection computation
/****************************************************************/

bool HeightMap_BinarySearch( float3 raydir, float3 ray_origin, float3 bmin, float3 bmax, sampler2D heightmapSampler, out float3 position, out float4 heightMapValue )
{

	float2 scale = 1/( abs(bmin.zy) + abs(bmax.zy) );
	float2 offset = ( ray_origin.zy - float2(bmin.z, bmax.y) );
	
	float2 texcoord = offset * float2(scale.x, -scale.y);
	float2 delta_uv = texcoord / offset;						
							

	raydir *= 0.5;
	position = ray_origin + raydir;
	texcoord.xy += raydir.zy * delta_uv;
	
	bool is_intersection = false;
	bool decrease = false;
	bool increase = false;
		
	for( int i=0; i<NumberOfIterations; i++ )	//binary search
	{		
		float height = tex2Dlod( heightmapSampler, float4(texcoord,0,0) ).a;
				
		raydir *= 0.5;
				
		if ( position.x < height ) 
		{	
			increase = true;
			position += raydir;
			texcoord.xy += raydir.zy * delta_uv;
		}
		else
		{	
			decrease = true;
			position -= raydir;
			texcoord.xy -= raydir.zy * delta_uv;
		}		
	}
		
	heightMapValue = tex2Dlod( heightmapSampler, float4(texcoord,0,0) );
	
	if( increase==true && decrease==true ) is_intersection = true;
	
	return is_intersection;
	
}




//****************************************************************/
//Ray - object height map impostor intersection calculation
/****************************************************************/

bool IntersectHeightMap( float3 raydir, inout float3 raypos, float3 objpos, float3 lookatfrom, float3 bmin, float3 bmax, sampler2D heightmapSampler, out float3 normal, bool computeRefraction )	
{	
	normal = float3(0,0,0);
	
	float3 lookat = lookatfrom - objpos;
	
	float3x3 TBN;
	TBN[0] = normalize( float3( lookat.x, 0, lookat.z ) );
	TBN[1] = float3(0, 1, 0);
	TBN[2] = cross( TBN[1], TBN[0] );
	
	//transforming ray origin and direction to the coordinate system of the plane of symmetry
	raypos = raypos - objpos;
	raypos = mul( raypos, TBN );						
	raydir = mul( raydir, TBN );		
	
	
	//first intersection with the AABB is inspected
	float d_boxnear, d_boxfar;
	if( !IntersectBoundingBox(raypos ,raydir, bmin, bmax, d_boxnear, d_boxfar) ) return false;
	d_boxnear = max(d_boxnear, 0);
	
	//intersection with the plane of symmetry
	float d_plane = -raypos.x / raydir.x;
	
	
	float d_start, d_stop;
	bool mirror = false;
	bool intersect_plane = true;		
	
	if( d_plane > d_boxfar || d_plane < d_boxnear ) 
	{
		d_start = d_boxnear;
		d_stop = d_boxfar;
		intersect_plane = false;		
	}
	else
	{
		d_start = d_boxnear;
		d_stop = d_plane;
	}
	
		
	float3 start = raypos + d_start*raydir;
	float3 stop = raypos + d_stop*raydir;
																				
	if( start.x < 0 )	//if 'start' and 'stop' are behind the plane 
	{
		start.x = -start.x;
		stop.x = -stop.x;
		mirror = true;
	}
	
	float3 init;
	if(computeRefraction)
	{			
		raydir = stop - start;
		init = start;
	}
	else
	{
		raydir = start - stop;
		init = stop;
	}
	
	float3 position;	
	float4 heightMapValue;
	
	//binary search
	bool intersection_found = HeightMap_BinarySearch( raydir, init, bmin, bmax, heightmapSampler, position, heightMapValue );
	
	
	bool no_intersection = false;
	if( ( heightMapValue.r==0 && heightMapValue.g==0 && heightMapValue.b==0 ) ||
		( !intersection_found && abs(position.x-heightMapValue.a)>0.01 ) )
		no_intersection = true;
	 
	if( intersect_plane && no_intersection )	//it is possible that there is an intersection with the piece behind the plane
	{
	
		mirror = true;
		
		if(computeRefraction)
			raydir.x = -raydir.x;
		else
			raydir.y = -raydir.y;
			
		raydir /= raydir.x;
		raydir *= abs(bmax.x);
	
		start = stop + raydir;
	
		if( start.y<bmin.y || start.y>bmax.y ||
			start.z<bmin.z || start.z>bmax.z )
		{
			raydir = normalize(raydir);
			IntersectBoundingBox(stop, raydir, bmin, bmax, d_boxnear, d_boxfar);
			start = stop + d_boxfar*raydir;
			raydir = start - stop; 
		}
		
		//binary search
		intersection_found = HeightMap_BinarySearch( raydir, stop, bmin, bmax, heightmapSampler, position, heightMapValue );
		
		
		if( !computeRefraction )
		{
			if( !( heightMapValue.r==0 && heightMapValue.g==0 && heightMapValue.b==0 ) && 
				(intersection_found || ( !intersection_found && abs(position.x-heightMapValue.a)<0.01 ) ) )
				no_intersection = false;
		}
		
	}
	
	
	if( !computeRefraction && no_intersection ) return false;
	
	
	if( mirror )
	{
		position.x = -position.x;
		heightMapValue.r = -heightMapValue.r;
	}
	
		
	raypos = ( position.x * TBN[0] + position.y * TBN[1] + position.z * TBN[2] ) + objpos;
	normal = normalize( heightMapValue.r * TBN[0] + heightMapValue.g * TBN[1] + heightMapValue.b * TBN[2] );
	
	return true;
	
}




/****************************************************************/
//This function traces rays through a refractive object. The object is represented by a height map impostor.
//It performs double refraction calculation. 
/****************************************************************/

bool RefractHeightMap( inout float3 raydir, inout float3 raypos, inout float3 normal, float3 objpos, float3 lookatfrom, float3 bmin, float3 bmax, sampler2D heightmapSampler )	
{	
	
	bool retValue = true;
	
	float3 refr_dir = refract(raydir, normal, IndexOfRefraction);					//1st refraction
	
	if(refr_dir.x==0 && refr_dir.y==0 && refr_dir.z==0)		//total reflection
	{
		refr_dir = reflect(raydir, normal);
		return true;	
	}
	
	//2nd refraction
	if( IntersectHeightMap( refr_dir, raypos, objpos, lookatfrom, bmin, bmax, heightmapSampler, normal, true ) )
	{
		raydir = refract(refr_dir, -normal, 1/IndexOfRefraction);
	
		if( raydir.x==0 && raydir.y==0 && raydir.z==0 ) raydir = reflect(refr_dir, -normal);
	}
	else retValue = false;
	
	return retValue;
	
}




/****************************************************************/
//This shader stores the incoming radiance from the sky in an environment distance impostor.
/****************************************************************/

void vsFullScreenQuad
(
	in float4 pos : POSITION,
	out float4 posout : POSITION,
	out float3 viewDir : TEXCOORD0
) 
{
	posout = pos;
	
    float4 hWorldPos = mul(pos, viewProjInverseMatrix);
	
	//homogeneus division
    hWorldPos /= hWorldPos.w;

    viewDir = hWorldPos.xyz - EyePos;
}

float4 psComputeEnvMap_background
(
	float4 pos : POSITION,
	float3 viewDir : TEXCOORD0
) : COLOR
{	
	return float4( texCUBE(skymap, viewDir).rgb, length(viewDir) );
}




/****************************************************************/
//This shader renders diffuse objects into an environment distance impostor.
//It stores the radiance incoming from the object
//and the distance between the object's surface point's position and the reference point.
/****************************************************************/

void vsComputeEnvMap_diffuse
(
	in float4 pos : POSITION,
	in float2 tex : TEXCOORD0,
	out float4 posout : POSITION,
	out float2 texout : TEXCOORD0,
	out float3 worldPosout : TEXCOORD1,
	out float3 viewDirout : TEXCOORD2
) 
{
	texout = tex;
	posout = mul( pos, modelViewProjMatrix );
	worldPosout = mul( pos, modelMatrix ).xyz;
	viewDirout = worldPosout - EyePos;
}

float4 psComputeEnvMap_diffuse
(
	float4 pos : POSITION,
	float2 tex : TEXCOORD0,
	float3 worldpos : TEXCOORD1,
	float3 viewDir : TEXCOORD2
) : COLOR
{	
	float3 retColor = tex2D(diffuseMapSampler, tex).rgb;		//diffuse color
	retColor += tex2D(diffuseMapFilter, tex).rgb;				//light map of the static environment		
	
	//light map of the dynamic environment
	if( id_moving > 0 ) retColor += tex2D(diffuseMapFilter_moving, tex).rgb;
	
	return float4(retColor, length(viewDir));		
}




/****************************************************************/
//This shader renders refractive and reflective chess pieces into an environment distance impostor.
//It stores the radiance incoming from the object
//and the distance between the object's surface point's position and the reference point.
/****************************************************************/
void vsComputeEnvMap
(
	in float4 pos : POSITION,
	in float3 normal : NORMAL,
	out float4 posout : POSITION,
	out float3 worldPosout : TEXCOORD0,
	out float3 normalout : TEXCOORD1,
	out float3 viewout : TEXCOORD2,
	out float3 envPosout : TEXCOORD3
) 
{
	posout = mul( pos, modelViewProjMatrix );
	worldPosout = mul( pos, modelMatrix ).xyz;
	normalout =	mul( modelMatrixInverse, float4(normal.xyz, 0.0) );
	viewout = worldPosout - EyePos;
	envPosout = worldPosout - selected_obj_position;
}

float4 psComputeEnvMap_reflect_refract
(
	float4 pos : POSITION,
	float3 worldpos : TEXCOORD0,
	float3 normal : TEXCOORD1,
	float3 view : TEXCOORD2,
	float3 envPos : TEXCOORD3
) : COLOR
{	
	float3 raydir = normalize(view);
	normal = normalize(normal);
	
	float3 reflect_color = float3(0,0,0);
	float3 refract_color = float3(0,0,0);
	
	//refrlection direction
	float3 reflect_dir = reflect(raydir, normal);
	float3 Hit_result = Hit(envPos, reflect_dir, envmap);
	reflect_color = texCUBE(envmap, Hit_result).rgb;
	
	//refraction direction
	float3 refract_dir = raydir;
	float3 refract_pos = worldpos;
	float3 refract_normal = normal;
	if( RefractHeightMap( refract_dir, refract_pos, refract_normal, selected_obj_position, EyePos, selected_obj_boundingMin, selected_obj_boundingMax, diffuseMapSampler )	)
	{
		Hit_result = Hit(refract_pos - selected_obj_position, refract_dir, envmap);
		refract_color = texCUBE(envmap, Hit_result).rgb;
	}
	
	float F = FresnelFactor + ( 1 - FresnelFactor ) * pow( 1 - dot( normal, -raydir ), 5 );
	
	return float4( lerp( refract_color , reflect_color, F ) + GetColor(id), length(view) );	
}




/****************************************************************/
//This shader creates the photon map.
//Pictures are taken from the position of the light source and rays are traced through the glass pieces.
//The shader stores the texture coordinates of the chessboard's surface points and the power of the photon hit. 
/****************************************************************/

void vsRenderPhotonMap
(
    in float4 pos : POSITION,
    in float3 normal : NORMAL,
    out float4 posout : POSITION,
    out float3 worldPosout : TEXCOORD0,
    out float3 normalout : TEXCOORD1
)
{  
    posout = mul( pos, modelViewProjMatrix );		
	worldPosout = mul( pos, modelMatrix ).xyz;
	normalout =	mul( modelMatrixInverse, float4(normal.xyz,0) );
}

float4 psRenderPhotonMap
(
	float4 pos : POSITION,
    float3 worldPos : TEXCOORD0,
    float3 normal   : TEXCOORD1
) : COLOR
{
	float a = -1;
	float3 hitSurf = float3(0,0,0);
	normal = normalize( normal ); 
		
		
	float3 refract_dir = LightDir;
	float3 refract_pos = worldPos;
	float3 refract_normal = normal;
	//we follow the ray through the glass piece
	if( RefractHeightMap( refract_dir, refract_pos, refract_normal, selected_obj_position, EyePos, selected_obj_boundingMin, selected_obj_boundingMax, diffuseMapSampler )	)
	{
		//intersection computation with the chessboard
		//the equation of the chessbard's plane is 'y=0'
	
		float rayparam = -refract_pos.y / refract_dir.y;		//we solve the equation 'origin.y+d*dir.y=0' with unknown variable 'd'
	
		if( rayparam >= 0 )		//there is an intersection with the plane
		{
		
			float3 hitPoint = refract_pos + rayparam * refract_dir;

			float2 minXZ = chesstable_data.xy;
			float2 maxXZ = chesstable_data.zw;
			
			//the intersection point must be on the chessboard
			if(hitPoint.x>=minXZ.x && hitPoint.z>=minXZ.y &&
			   hitPoint.x<=maxXZ.x && hitPoint.z<=maxXZ.y)
			{
				float2 uv = (-hitPoint.xz + maxXZ.xy) / (maxXZ.xy - minXZ.xy);
							
				float F = FresnelFactor + ( 1 - FresnelFactor ) * pow( 1-dot(normal, -LightDir), 5 );
				
				hitSurf = float3(uv,0);
				a = (1-F) * Power;
			}
			
		}
		
	}
			
	return float4( hitSurf, a );
}



/****************************************************************/
// This shader renders caustics into the light map.
/****************************************************************/

void vsRenderPhotonHit
(
    in float4 pos : POSITION,
    in float4 tex: TEXCOORD0,
    out float4 posout : POSITION,
    out float2 texout : TEXCOORD0,
    out float powerout : TEXCOORD1
)
{
	float2 size = tex.xy;
	float2 ph_uv = pos.xy;
	texout = tex.zw;

	//reads back the photon hit's UV position
	float4 ph = tex2Dlod(causticmap, float4( ph_uv, 0, 0));

    posout = float4( 2 * ph.r - 1 + size.x + HALF,
					-2 * ph.g + 1 + size.y - HALF,
					 0 , 1);	
    
    if ( ph.a <= 0 ) posout.z = -100;	//invalid photon hit
	
	powerout = ph.w;
}

float4 psRenderPhotonHit
(
	float4 pos : POSITION,
	float2 tex : TEXCOORD0,
    float power : TEXCOORD1
) : COLOR
{	
	float4 retColor;
	
	//the gaussian weight of a snippet.
	float4 w = tex2D( PowerOfSnippetMap, tex );
	
	//the color of the snippet's pixel.
	retColor = w * power * CausticsIntensity;
	
	return float4( retColor.xyz, 1 );
}




/****************************************************************/
//This shader calculates the shadow map.
/****************************************************************/

void vsShadowMap
(
	in float4 pos : POSITION,
    out float4 posout : POSITION,
    out float4 screenPosout : TEXCOORD0
)
{
    posout = mul( pos, modelViewProjMatrix );
    screenPosout = posout / posout.w;
}

float4 psShadowMap
(
	float4 pos : POSITION,
	float4 screenPos : TEXCOORD0
) : COLOR
{
    return float4(screenPos.z, 0, 0, 1);
}




/****************************************************************/
// This shader renders shadow into the light map.
/****************************************************************/

void vsRenderShadow
(
	in float4 pos : POSITION,
	in float2 tex : TEXCOORD0,
    in float3 normal : NORMAL,
    out float4 posout : POSITION,
    out float2 texout: TEXCOORD0,
    out float4 lightPosout : TEXCOORD1
)
{
    posout = float4( float2(2, -2) * tex + float2(-1-HALF, 1+HALF), 0, 1);
    
    texout = tex;
    lightPosout = mul( pos, LightModelViewProjTexBias );    
}

float4 psRenderShadow
(
	float2 tex  : TEXCOORD0,
    float4 lightPos : TEXCOORD1
) : COLOR
{
	float4 retColor = float4(-0.1,-0.1,-0.04,0.8);	
	float isShadow = 1;
	
	float3 projPos = lightPos.xyz / lightPos.w;
	
	if( tex2D( shadowMap, projPos.xy ).x > projPos.z  ) isShadow = 0;
	
    return isShadow * retColor;
}




/****************************************************************/
//MapRender technique
/****************************************************************/

technique MapRender
{
	pass RefrMapCreation
	{
		CullMode = CW;
		VertexShader = compile vs_3_0 vsComputeRefractor();
		PixelShader = compile ps_3_0 psComputeRefractor();
	}
	pass EnvMapCreation_background
	{	
		CullMode = CW;
		VertexShader = compile vs_3_0 vsFullScreenQuad();
		PixelShader = compile ps_3_0 psComputeEnvMap_background();
	}
	pass EnvMapCreation_diffuse
	{	CullMode = CCW;
		VertexShader = compile vs_3_0 vsComputeEnvMap_diffuse();
		PixelShader = compile ps_3_0 psComputeEnvMap_diffuse();
	}
	pass EnvMapCreation_reflection_and_refraction
	{
		VertexShader = compile vs_3_0 vsComputeEnvMap();
		PixelShader = compile ps_3_0 psComputeEnvMap_reflect_refract();
	}
	pass RenderPhotonMap
	{
		VertexShader = compile vs_3_0 vsRenderPhotonMap();
		PixelShader = compile ps_3_0 psRenderPhotonMap();
	}
	pass RenderPhotonHit
	{	
		AlphaBlendEnable = true;
		SrcBlend = One;
		DestBlend = One;
		ZWriteEnable = false;
		VertexShader = compile vs_3_0 vsRenderPhotonHit();
		PixelShader = compile ps_3_0 psRenderPhotonHit();
	}
	pass HeightMapCreation
	{
		VertexShader = compile vs_3_0 vsComputeRefractor();
		PixelShader = compile ps_3_0 psComputeHeightMap();
	}
	pass ShadowMap
	{
		VertexShader = compile vs_3_0 vsShadowMap();
		PixelShader = compile ps_3_0 psShadowMap();
	}
	pass RenderShadow
	{
		AlphaBlendEnable = true;
		SrcBlend = One;
		DestBlend = One;
		ZWriteEnable = false;
		VertexShader = compile vs_3_0 vsRenderShadow();
		PixelShader = compile ps_3_0 psRenderShadow();
	}
}
















/****************************************************************/
//This function examines whether the moving piece is seen in a given direction.
/****************************************************************/

float4 computeColor_Moving_Object( float3 worldPos, float3 envPos, float3 raydir )
{

	float4 returnColor = float4(0,0,0,0);
	
	float distanceFromStatic = 0;
	float distanceFromMoving = 0;
	
	
	//intersection with the static environment
	float3 hitDirection = Hit(envPos, raydir, envmap);
	returnColor = texCUBE(envmap, hitDirection);
	distanceFromStatic = distance( hitDirection*returnColor.a/length(hitDirection), envPos);

	
	//if the ray intersects the moving piece...
	float3 hitPos = worldPos, hitNormal;
	if( IntersectHeightMap( raydir, hitPos, moving_obj_position, selected_obj_position, moving_obj_boundingMin, moving_obj_boundingMax, diffuseMapFilter, hitNormal, false ) )	
	{
		
		distanceFromMoving = distance(worldPos, hitPos);
				
		if( distanceFromMoving < distanceFromStatic )	//if the moving piece is nearer than the static environment...
		{	
			//we continue the search at the reflection direction by querying the moving piece's environment impostor.  
			hitDirection = Hit(hitPos - moving_obj_position, reflect(raydir, hitNormal), envmap_moving);
		
			float3 refract_dir = raydir;
			float3 refract_normal = hitNormal;
			//To obtain the incoming radiance from the refraction direction we trace the ray through the moving piece.
			if( RefractHeightMap( refract_dir, hitPos, refract_normal, moving_obj_position, selected_obj_position, moving_obj_boundingMin, moving_obj_boundingMax, diffuseMapFilter )	)
			{
				//we query the moving piece's environment impostor again. 		
				float3 hitDirection2 = Hit(hitPos - moving_obj_position, refract_dir, envmap_moving);
			
				float F = FresnelFactor + ( 1 - FresnelFactor ) * pow( 1 - dot( hitNormal, -raydir ), 5 );
			
				//The final color will be the combination of the two radiance value.
				returnColor = lerp( texCUBE(envmap_moving, hitDirection2), texCUBE(envmap_moving, hitDirection), F ) + float4(GetColor(id_moving), 0);
				
			}
			
		}
			
	}
	
	
	//If the static environment is nearer, we examine whether the intersection point is on the chessboard.
	//In this case we have to modulate the color of the chessboard with the moving piece's light map.
	if( distanceFromMoving==0 || distanceFromStatic < distanceFromMoving )
	{	
		//Intersection computation with the chessboard.
		//The equation of the chessbard's plane is 'y=0'.
		//We solve the equation 'origin.y+d*dir.y=0' with unknown variable 'd'
		
		float distanceFromChessboard = -worldPos.y / raydir.y;
	
		if( distanceFromChessboard > 0 )		//there is an intersection with the chessboard's plane
		{
			if( abs( distanceFromStatic - distanceFromChessboard )< 1.2 )	//if the distance between the intersection points is negligible	
			{
				float2 minXZ = chesstable_data.xy;
				float2 maxXZ = chesstable_data.zw;
				
				float3 hit = worldPos + distanceFromChessboard * raydir;
				float2 uv = (-hit.xz + maxXZ.xy) / (maxXZ.xy - minXZ.xy);
							
				returnColor.rgb += tex2D(diffuseMapFilter_moving, uv).rgb;
			}	
		}
	}
	
	return returnColor;
	
}




/****************************************************************/
//This shader traces rays through a glass piece.
//We use this shader if the piece's entire environment is static.
/****************************************************************/

void vsRender
(
	in float4 pos : POSITION,
	in float3 normal : NORMAL,
	out float4 posout : POSITION,
	out float3 worldPosout : TEXCOORD0,
	out float3 normalout : TEXCOORD1,
	out float3 viewout : TEXCOORD2,
	out float3 envPosout : TEXCOORD3
) 
{
	posout = mul( pos, modelViewProjMatrix );
	worldPosout = mul( pos, modelMatrix ).xyz;
	normalout =	mul( modelMatrixInverse, float4(normal.xyz, 0.0) );
	viewout = worldPosout.xyz-EyePos;
	envPosout = worldPosout - selected_obj_position;
}

float4 psRender_static
(
	float4 pos : POSITION,
	float3 worldPos : TEXCOORD0,
	float3 normal : TEXCOORD1,
	float3 view : TEXCOORD2,
	float3 envPos : TEXCOORD3
) : COLOR
{	
	float4 reflect_color = float4(0,0,0,0);
	float4 refract_color = float4(0,0,0,0);
	
	view = normalize(view);
	normal = normalize(normal);
		
		
	//reflection direction 
	float3 reflect_dir = reflect(view, normal);
	float3 Hit_result = Hit(envPos, reflect_dir, envmap);	//we search in the piece's environment impostor
	reflect_color = texCUBE(envmap, Hit_result);
	
	
	//refraction direction
	float3 refract_dir = view;
	float3 refract_pos = worldPos;
	float3 refract_normal = normal;
	
	//One search is needed in the height map impostor to trace the ray through the object.
	if( RefractHeightMap( refract_dir, refract_pos, refract_normal, selected_obj_position, EyePos, selected_obj_boundingMin, selected_obj_boundingMax, diffuseMapSampler )	)
	{	
		Hit_result = Hit(refract_pos - selected_obj_position, refract_dir, envmap);			//we search in the piece's environment impostor again
		refract_color = texCUBE(envmap, Hit_result);
	}
	
	
	float F = FresnelFactor + ( 1 - FresnelFactor ) * pow( 1 - dot( normal, -view ), 5 );
	
	return lerp( refract_color , reflect_color, F ) + float4(GetColor(id), 0);	
}




/****************************************************************/
//This shader traces rays through a glass piece.
//We use this shader if the piece's environment is separated into static and dynamic parts.
/****************************************************************/

float4 psRender_static_with_moving_obj
(
	float4 pos : POSITION,
	float3 worldPos : TEXCOORD0,
	float3 normal : TEXCOORD1,
	float3 view : TEXCOORD2,
	float3 envPos : TEXCOORD3
) : COLOR
{	
	float4 reflect_color = float4(0,0,0,0);
	float4 refract_color = float4(0,0,0,0);
	
	view = normalize(view);
	normal = normalize(normal);
	
		
	//reflection direction 
	
	//We eximine whether the static environment is occluded by the moving piece in the reflection direction.
	reflect_color = computeColor_Moving_Object( worldPos, envPos, reflect(view, normal) );
	
	
	//refraction direction 
	
	float3 refract_dir = view;
	float3 refract_pos = worldPos;
	float3 refract_normal = normal;
	
	//One search is needed in the height map impostor to trace the ray through the object.
	if( RefractHeightMap( refract_dir, refract_pos, refract_normal, selected_obj_position, EyePos, selected_obj_boundingMin, selected_obj_boundingMax, diffuseMapSampler )	)
	{
		//We eximine whether the static environment is occluded by the moving piece in the refraction direction.
		refract_color = computeColor_Moving_Object( refract_pos, refract_pos - selected_obj_position, refract_dir );
	}
	
	float F = FresnelFactor + ( 1 - FresnelFactor ) * pow( 1 - dot( normal, -view ), 5 );
	return lerp( refract_color , reflect_color, F ) + float4(GetColor(id), 0);		
}




/****************************************************************/
//This shader reads the sky cube map in the view direction.
/****************************************************************/

float4 psSkyRender
(
	float4 pos : POSITION,
	float3 viewDir : TEXCOORD0
) : COLOR
{
	return texCUBE(skymap,viewDir);
}




/****************************************************************/
//This shader renders diffuse objects.
//It modulates the diffuse color with the light light maps.
/****************************************************************/

void vsDiffuse
(
	in float4 pos : POSITION,
	in float2 tex : TEXCOORD0,
	out float4 posout : POSITION,
	out float2 texout : TEXCOORD0
) 
{
	texout = tex;
	posout = mul( pos, modelViewProjMatrix );
}

float4 psDiffuse
(
	float4 pos : POSITION,
	float2 tex : TEXCOORD0
) : COLOR
{
	float3 retColor = tex2D(diffuseMapSampler, tex).rgb;
	
	if( id >= 0 )
	{
		retColor += tex2D(diffuseMapFilter, tex).rgb;

		if( id_moving > 0 ) retColor += tex2D(diffuseMapFilter_moving, tex).rgb;
	}
	
	return float4(retColor, 1);	
}




technique render
{
	pass sky_render
	{
		CullMode = CW;
		VertexShader = compile vs_3_0 vsFullScreenQuad();
		PixelShader = compile ps_3_0 psSkyRender();
	}
	pass approx_raytrace_static
	{
		VertexShader = compile vs_3_0 vsRender();
		PixelShader = compile ps_3_0 psRender_static();
	}
	pass approx_raytrace_static_with_moving_obj
	{
		VertexShader = compile vs_3_0 vsRender();
		PixelShader = compile ps_3_0 psRender_static_with_moving_obj();
	}
	pass diffuse_render
	{
		VertexShader = compile vs_3_0 vsDiffuse();
		PixelShader = compile ps_3_0 psDiffuse();
	}	
}