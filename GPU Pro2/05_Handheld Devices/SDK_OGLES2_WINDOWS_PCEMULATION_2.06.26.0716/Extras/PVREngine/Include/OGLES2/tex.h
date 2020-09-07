/******************************************************************************

 @File         tex.h

 @Title        PVREngine main header file for OGLES2 API

 @Version      

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     ANSI compatible

 @Description  A default PFX to be used with the PVREngine when no other PFX is
               specified.

******************************************************************************/

#ifndef _TEX_H_
#define _TEX_H_

const char texshader[] =
"[HEADER]\n"
"	VERSION		01.00.00.00\n"
"	DESCRIPTION Texturing with a single diffuse point light\n"
"	COPYRIGHT	Img Tec\n"
"[/HEADER]\n"

"[TEXTURES]\n"
"	FILE basemap		Flat.pvr	LINEAR-LINEAR-LINEAR\n"
"[/TEXTURES]\n"

"[EFFECT]\n"
"	NAME 	BasicDiffuse\n"

	// GLOBALS UNIFORMS
"	UNIFORM myMVPMatrix 	WORLDVIEWPROJECTION\n"
"	UNIFORM	basemap			TEXTURE0\n"

	// ATTRIBUTES
"	ATTRIBUTE 	myVertex	POSITION\n"
"	ATTRIBUTE	myUV		UV\n"

"	VERTEXSHADER MyVertexShader\n"
"	FRAGMENTSHADER MyFragmentShader\n"
"	TEXTURE 0 basemap\n"

"[/EFFECT]\n"

"[VERTEXSHADER]\n"
"	NAME 		MyVertexShader\n"

	// LOAD GLSL AS CODE
"	[GLSL_CODE]\n"
"	attribute highp vec3	myVertex;\n"
"	attribute mediump vec2	myUV;\n"

"	uniform highp mat4	myMVPMatrix;\n"

"	varying mediump vec2	texCoordinate;\n"


"		void main(void)\n"
"		{\n"
"			gl_Position = myMVPMatrix * vec4(myVertex,1.0);\n"
"			texCoordinate = myUV.st;\n"
"		}\n"
"	[/GLSL_CODE]\n"
"[/VERTEXSHADER]\n"

"[FRAGMENTSHADER]\n"
"	NAME 		MyFragmentShader\n"

	// LOAD GLSL AS CODE
"	[GLSL_CODE]\n"
"		uniform sampler2D 		basemap;\n"
"		varying mediump vec2	texCoordinate;\n"

"		void main (void)\n"
"		{\n"
"			gl_FragColor =   texture2D(basemap, texCoordinate);\n"
"		}\n"
"	[/GLSL_CODE]\n"
"[/FRAGMENTSHADER]\n";

#endif	// _TEX_H_

