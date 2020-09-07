// This file was created by Filewrap 1.1
// Little endian mode
// DO NOT EDIT

#include "../PVRTMemoryFileSystem.h"

// using 32 bit to guarantee alignment.
#ifndef A32BIT
 #define A32BIT static const unsigned int
#endif

// ******** Start: SkyboxVShader.vsh ********

// File data
static const char _SkyboxVShader_vsh[] = 
	"attribute mediump vec3 inVertex;\r\n"
	"\r\n"
	"uniform mediump mat4 ModelMatrix;\r\n"
	"uniform mediump mat4 ModelViewMatrix;\r\n"
	"uniform mediump mat4 MVPMatrix;\r\n"
	"#ifdef ENABLE_FOG_DEPTH\r\n"
	"uniform mediump float WaterHeight;\t\t//Assume water always lies on the y-axis\r\n"
	"#endif\r\n"
	"#ifdef ENABLE_DISCARD_CLIP\r\n"
	"uniform bool ClipPlaneBool;\r\n"
	"uniform mediump vec4 ClipPlane;\r\n"
	"#endif\r\n"
	"\r\n"
	"varying mediump vec3 EyeDir;\r\n"
	"#ifdef ENABLE_FOG_DEPTH\r\n"
	"varying mediump float VertexDepth;\r\n"
	"#endif\r\n"
	"#ifdef ENABLE_DISCARD_CLIP\r\n"
	"varying highp float ClipDist;\r\n"
	"#endif\r\n"
	"\r\n"
	"void main()\r\n"
	"{\r\n"
	"\tEyeDir = -inVertex;\r\n"
	"\tgl_Position = MVPMatrix * vec4(inVertex, 1.0);\r\n"
	"\t\r\n"
	"\t#ifdef ENABLE_DISCARD_CLIP\r\n"
	"\t\t// Compute the distance between the vertex and clipping plane (in world space coord system)\r\n"
	"\t\tmediump vec4 vVertexView = ModelMatrix * vec4(inVertex.xyz,1.0);\r\n"
	"\t\tClipDist = dot(vVertexView, ClipPlane);\r\n"
	"\t#endif\r\n"
	"\t\r\n"
	"\t#ifdef ENABLE_FOG_DEPTH\r\n"
	"\t\t// Calculate the vertex's distance under water surface. This assumes clipping has removed all objects above the water\r\n"
	"\t\tmediump float vVertexHeight = (ModelMatrix * vec4(inVertex,1.0)).y;\r\n"
	"\t\tVertexDepth = WaterHeight - vVertexHeight;\r\n"
	"\t#endif\r\n"
	"}";

// Register SkyboxVShader.vsh in memory file system at application startup time
static CPVRTMemoryFileSystem RegisterFile_SkyboxVShader_vsh("SkyboxVShader.vsh", _SkyboxVShader_vsh, 1133);

// ******** End: SkyboxVShader.vsh ********

// This file was created by Filewrap 1.1
// Little endian mode
// DO NOT EDIT

#include "../PVRTMemoryFileSystem.h"

// using 32 bit to guarantee alignment.
#ifndef A32BIT
 #define A32BIT static const unsigned int
#endif

// ******** Start: SkyboxVShader.vsc ********

// File data
A32BIT _SkyboxVShader_vsc[] = {
0x10fab438,0x219250b6,0x30050100,0x2101,0xa9142cc2,0x0,0x0,0x6a020000,0x0,0x4000000,0x0,0x9000000,0x10003,0x0,0x20100,0x0,0x1,0x84010000,0x55535020,0x17,0x178,0x1,0x0,0x80c,0x0,0x2,0x79,0x0,0x8,0x0,0xffffffff,0x0,0x76000a,0xffff,0x10003,0x0,0x100000,0x0,0x0,0x0,0x0,0xfffc0000,0x0,0x0,0x0,0x20000,0xffffffff,0x80000,0x20003,0x4,0x20000,0x5,0x20010,0x10006,0x20000,0x10007,0x20010,0x20008,0x20000,0x20009,0x20010,0x3000a,0x20000,0x3000b,
0x20010,0x4000c,0x20000,0x4000d,0x20010,0x5000e,0x20000,0x5000f,0x20010,0x60010,0x20000,0x60011,0x20010,0x70012,0x20000,0x70013,0x70010,0x40000,0x20000,0x2,0x1000a,0x80018001,0x80018001,0x0,0x0,0x4,0x10000,0x10001,0x10000,0x10001,0x1,0xfd100000,0x50801a30,0xa32085,0xf0002a55,0x38ab0604,0xf0006c59,0x38820604,0xf000ae5d,0x38820604,0x700f1a60,0xe21001,0xf0002b57,0x38ab0606,0xf0006d5b,0x38820606,0xf000af5f,0x38820606,0x704f1a62,0xe21001,0x6,0x803f1400,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,
0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x4000000,0x44657945,0x7269,0x50400,0x100,0x3000001,0x700,0x65566e69,0x78657472,0x4000000,0x1000004,0x10000,0x7000400,0x564d0000,0x74614d50,0x786972,0x3160000,0x10000,0x4000100,0xffff10,0x5f6c6700,0x69736f50,0x6e6f6974,0x5010000,0x1000005,0x10000,0xf000400,0x0,
};

// Register SkyboxVShader.vsc in memory file system at application startup time
static CPVRTMemoryFileSystem RegisterFile_SkyboxVShader_vsc("SkyboxVShader.vsc", _SkyboxVShader_vsc, 650);

// ******** End: SkyboxVShader.vsc ********

