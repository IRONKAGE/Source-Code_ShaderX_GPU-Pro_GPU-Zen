attribute mediump vec3 inVertex;

uniform mediump mat4 ModelMatrix;
uniform mediump mat4 ModelViewMatrix;
uniform mediump mat4 MVPMatrix;
#ifdef ENABLE_FOG_DEPTH
uniform mediump float WaterHeight;		//Assume water always lies on the y-axis
#endif
#ifdef ENABLE_DISCARD_CLIP
uniform bool ClipPlaneBool;
uniform mediump vec4 ClipPlane;
#endif

varying mediump vec3 EyeDir;
#ifdef ENABLE_FOG_DEPTH
varying mediump float VertexDepth;
#endif
#ifdef ENABLE_DISCARD_CLIP
varying highp float ClipDist;
#endif

void main()
{
	EyeDir = -inVertex;
	gl_Position = MVPMatrix * vec4(inVertex, 1.0);
	
	#ifdef ENABLE_DISCARD_CLIP
		// Compute the distance between the vertex and clipping plane (in world space coord system)
		mediump vec4 vVertexView = ModelMatrix * vec4(inVertex.xyz,1.0);
		ClipDist = dot(vVertexView, ClipPlane);
	#endif
	
	#ifdef ENABLE_FOG_DEPTH
		// Calculate the vertex's distance under water surface. This assumes clipping has removed all objects above the water
		mediump float vVertexHeight = (ModelMatrix * vec4(inVertex,1.0)).y;
		VertexDepth = WaterHeight - vVertexHeight;
	#endif
}