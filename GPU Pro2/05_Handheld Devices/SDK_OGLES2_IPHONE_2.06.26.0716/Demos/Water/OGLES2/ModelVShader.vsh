#define ENABLE_TEXTURE
attribute highp vec3 	inVertex;
attribute highp vec3	inNormal;
attribute highp vec2	inTexCoord;

uniform highp mat4		MVPMatrix;
uniform mediump vec3	LightDirection;
#ifdef ENABLE_FOG_DEPTH
uniform highp mat4		ModelMatrix;
uniform mediump float	WaterHeight;		//Assume water always lies on the y-axis
#endif

#ifdef ENABLE_LIGHTING
	varying lowp float		LightIntensity;	
#endif
#ifdef ENABLE_TEXTURE
	varying mediump vec2 	TexCoord;
#endif
#ifdef ENABLE_FOG_DEPTH
	varying mediump float	VertexDepth;
#endif

void main()
{
	// Convert each vertex into projection-space and output the value
	highp vec4 vInVertex = vec4(inVertex, 1.0);
	gl_Position = MVPMatrix * vInVertex;
	
	#ifdef ENABLE_TEXTURE
		TexCoord = inTexCoord;
	#endif
	
	#ifdef ENABLE_FOG_DEPTH
		// Calculate the vertex's distance under water surface. This assumes clipping has removed all objects above the water
		mediump float vVertexHeight = (ModelMatrix * vec4(inVertex,1.0)).y;
		VertexDepth = WaterHeight - vVertexHeight;
	#endif
	
	#ifdef ENABLE_LIGHTING
		// Simple diffuse lighting in model space
		LightIntensity = 0.3 + clamp(dot(inNormal, -LightDirection),0.0,1.0);	// 0.5 is ambient light
	#endif
}
