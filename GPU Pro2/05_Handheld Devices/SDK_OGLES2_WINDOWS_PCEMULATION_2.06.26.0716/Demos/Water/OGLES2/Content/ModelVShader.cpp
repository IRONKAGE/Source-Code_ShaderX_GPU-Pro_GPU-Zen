// This file was created by Filewrap 1.1
// Little endian mode
// DO NOT EDIT

#include "../PVRTMemoryFileSystem.h"

// using 32 bit to guarantee alignment.
#ifndef A32BIT
 #define A32BIT static const unsigned int
#endif

// ******** Start: ModelVShader.vsh ********

// File data
static const char _ModelVShader_vsh[] = 
	"#define ENABLE_TEXTURE\r\n"
	"attribute highp vec3 \tinVertex;\r\n"
	"attribute highp vec3\tinNormal;\r\n"
	"attribute highp vec2\tinTexCoord;\r\n"
	"\r\n"
	"uniform highp mat4\t\tMVPMatrix;\r\n"
	"uniform mediump vec3\tLightDirection;\r\n"
	"#ifdef ENABLE_FOG_DEPTH\r\n"
	"uniform highp mat4\t\tModelMatrix;\r\n"
	"uniform mediump float\tWaterHeight;\t\t//Assume water always lies on the y-axis\r\n"
	"#endif\r\n"
	"\r\n"
	"#ifdef ENABLE_LIGHTING\r\n"
	"\tvarying lowp float\t\tLightIntensity;\t\r\n"
	"#endif\r\n"
	"#ifdef ENABLE_TEXTURE\r\n"
	"\tvarying mediump vec2 \tTexCoord;\r\n"
	"#endif\r\n"
	"#ifdef ENABLE_FOG_DEPTH\r\n"
	"\tvarying mediump float\tVertexDepth;\r\n"
	"#endif\r\n"
	"\r\n"
	"void main()\r\n"
	"{\r\n"
	"\t// Convert each vertex into projection-space and output the value\r\n"
	"\thighp vec4 vInVertex = vec4(inVertex, 1.0);\r\n"
	"\tgl_Position = MVPMatrix * vInVertex;\r\n"
	"\t\r\n"
	"\t#ifdef ENABLE_TEXTURE\r\n"
	"\t\tTexCoord = inTexCoord;\r\n"
	"\t#endif\r\n"
	"\t\r\n"
	"\t#ifdef ENABLE_FOG_DEPTH\r\n"
	"\t\t// Calculate the vertex's distance under water surface. This assumes clipping has removed all objects above the water\r\n"
	"\t\tmediump float vVertexHeight = (ModelMatrix * vec4(inVertex,1.0)).y;\r\n"
	"\t\tVertexDepth = WaterHeight - vVertexHeight;\r\n"
	"\t#endif\r\n"
	"\t\r\n"
	"\t#ifdef ENABLE_LIGHTING\r\n"
	"\t\t// Simple diffuse lighting in model space\r\n"
	"\t\tLightIntensity = 0.3 + clamp(dot(inNormal, -LightDirection),0.0,1.0);\t// 0.5 is ambient light\r\n"
	"\t#endif\r\n"
	"}\r\n";

// Register ModelVShader.vsh in memory file system at application startup time
static CPVRTMemoryFileSystem RegisterFile_ModelVShader_vsh("ModelVShader.vsh", _ModelVShader_vsh, 1239);

// ******** End: ModelVShader.vsh ********

// This file was created by Filewrap 1.1
// Little endian mode
// DO NOT EDIT

#include "../PVRTMemoryFileSystem.h"

// using 32 bit to guarantee alignment.
#ifndef A32BIT
 #define A32BIT static const unsigned int
#endif

// ******** Start: ModelVShader.vsc ********

// File data
A32BIT _ModelVShader_vsc[] = {
0x10fab438,0x766ff3c9,0x30050100,0x2101,0xa9142cc2,0x0,0x0,0x7e020000,0x0,0x4000000,0x0,0x9000000,0x2,0x0,0x20000,0x0,0x0,0x7a010000,0x55535020,0x17,0x16e,0x1,0x0,0x80c,0x0,0x2,0x79,0x0,0x8,0x0,0xffffffff,0x0,0x76000a,0xffff,0x6,0x0,0x100000,0x0,0x0,0x0,0x0,0xfffc0000,0x0,0x0,0x0,0x20000,0xffffffff,0x100000,0x10006,0x4,0x10000,0x10005,0x10000,0x20006,0x10000,0x30007,0x10000,0x40008,0x10000,0x50009,0x10000,0x6000a,0x10000,0x7000b,
0x10000,0x8000c,0x10000,0x9000d,0x10000,0xa000e,0x10000,0xb000f,0x10000,0xc0010,0x10000,0xd0011,0x10000,0xe0012,0x10000,0xf0013,0x370000,0x40000,0x20000,0x2,0x9,0x80018001,0x80018001,0x0,0x0,0x0,0x10001,0x1,0x10001,0x2000001,0x1001a080,0x50b28a1,0x606f060,0x470f38ab,0x606f060,0x89133882,0x606f060,0x1a163882,0x1001701f,0x60d00a2,0x606f000,0x481138ab,0x606f000,0x8a153882,0x606f000,0x1a183882,0x1001705f,0x600a2,0x14000000,0x803f,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,
0x0,0x0,0x0,0x0,0x0,0x6e690500,0x74726556,0x7865,0x40400,0x100,0x4000001,0x700,0x505f6c67,0x7469736f,0x6e6f69,0x5050100,0x10000,0x100,0xf0004,0x50564d00,0x7274614d,0x7869,0x31600,0x100,0x10040001,0xffff,0x43786554,0x64726f6f,0x3000000,0x1000005,0x10000,0x3000200,0x6e690000,0x43786554,0x64726f6f,0x3000000,0x1000004,0x10000,0x3000404,0x0,
};

// Register ModelVShader.vsc in memory file system at application startup time
static CPVRTMemoryFileSystem RegisterFile_ModelVShader_vsc("ModelVShader.vsc", _ModelVShader_vsc, 670);

// ******** End: ModelVShader.vsc ********

