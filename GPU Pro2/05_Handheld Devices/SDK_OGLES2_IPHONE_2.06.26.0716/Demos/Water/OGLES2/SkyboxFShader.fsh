uniform samplerCube CubeMap;

#ifdef ENABLE_FOG_DEPTH
uniform lowp vec3 FogColour;
uniform mediump float RcpMaxFogDepth;
#endif
#ifdef ENABLE_DISCARD_CLIP
uniform bool ClipPlaneBool;
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
	#ifdef ENABLE_DISCARD_CLIP
		// Reject fragments behind the clip plane
		if(ClipDist < 0.0)
		{
			discard; // Too slow for hardware. Left as an example of how not to do this!
		}
	#endif
	
	#ifdef ENABLE_FOG_DEPTH
		// Mix the object's colour with the fogging colour based on fragment's depth
		lowp vec3 vFragColour = textureCube(CubeMap, EyeDir).rgb;
		
		// Test depth
		lowp float fFogBlend = clamp(VertexDepth * RcpMaxFogDepth, 0.0, 1.0);
		vFragColour.rgb = mix(vFragColour.rgb, FogColour.rgb, fFogBlend);
			
		gl_FragColor = vec4(vFragColour.rgb, 1.0);
	#else
		gl_FragColor = textureCube(CubeMap, EyeDir);
		
	#endif
}