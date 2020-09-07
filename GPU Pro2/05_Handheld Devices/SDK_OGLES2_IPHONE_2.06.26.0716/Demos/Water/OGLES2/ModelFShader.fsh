#define ENABLE_TEXTURE
#ifdef ENABLE_TEXTURE
uniform sampler2D		ModelTexture;
#endif

#ifdef ENABLE_FOG_DEPTH
uniform lowp vec3 		FogColour;
uniform mediump float 	RcpMaxFogDepth;
#endif

#ifdef ENABLE_LIGHTING
	varying lowp float		LightIntensity;
#endif
#ifdef ENABLE_TEXTURE
	varying mediump vec2   	TexCoord;
#endif
#ifdef ENABLE_FOG_DEPTH
	varying mediump float 	VertexDepth;
#endif

void main()
{	
	#ifdef ONLY_ALPHA
		gl_FragColor = vec4(vec3(0.5),0.0);
	#else
		#ifdef ENABLE_TEXTURE
			#ifdef ENABLE_FOG_DEPTH		
				// Mix the object's colour with the fogging colour based on fragment's depth
				lowp vec3 vFragColour = texture2D(ModelTexture, TexCoord).rgb;
				
				// Perform depth test and clamp the values
				lowp float fFogBlend = clamp(VertexDepth * RcpMaxFogDepth, 0.0, 1.0);
				
				#ifdef ENABLE_LIGHTING
					vFragColour.rgb = mix(vFragColour.rgb * LightIntensity, FogColour.rgb, fFogBlend);
				#else
					vFragColour.rgb = mix(vFragColour.rgb, FogColour.rgb, fFogBlend);
				#endif
				gl_FragColor = vec4(vFragColour,1.0);
			#else
				#ifdef ENABLE_LIGHTING
					gl_FragColor = vec4(texture2D(ModelTexture, TexCoord).rgb * LightIntensity, 1.0);
				#else
					gl_FragColor = vec4(texture2D(ModelTexture, TexCoord).rgb, 1.0);
				#endif
			#endif
		#else
			// Solid colour is used instead of texture colour
			#ifdef ENABLE_LIGHTING
				gl_FragColor = vec4(vec3(0.3,0.3,0.3)* LightIntensity, 1.0);
			#else
				gl_FragColor = vec4(vec3(0.3,0.3,0.3), 1.0);	
			#endif
		#endif
	#endif
}
