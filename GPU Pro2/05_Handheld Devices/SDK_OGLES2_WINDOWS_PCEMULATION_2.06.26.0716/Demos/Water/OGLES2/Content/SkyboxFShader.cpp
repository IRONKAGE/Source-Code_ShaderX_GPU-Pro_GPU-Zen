// This file was created by Filewrap 1.1
// Little endian mode
// DO NOT EDIT

#include "../PVRTMemoryFileSystem.h"

// using 32 bit to guarantee alignment.
#ifndef A32BIT
 #define A32BIT static const unsigned int
#endif

// ******** Start: SkyboxFShader.fsh ********

// File data
static const char _SkyboxFShader_fsh[] = 
	"uniform samplerCube CubeMap;\r\n"
	"\r\n"
	"#ifdef ENABLE_FOG_DEPTH\r\n"
	"uniform lowp vec3 FogColour;\r\n"
	"uniform mediump float RcpMaxFogDepth;\r\n"
	"#endif\r\n"
	"#ifdef ENABLE_DISCARD_CLIP\r\n"
	"uniform bool ClipPlaneBool;\r\n"
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
	"\t#ifdef ENABLE_DISCARD_CLIP\r\n"
	"\t\t// Reject fragments behind the clip plane\r\n"
	"\t\tif(ClipDist < 0.0)\r\n"
	"\t\t{\r\n"
	"\t\t\tdiscard; // Too slow for hardware. Left as an example of how not to do this!\r\n"
	"\t\t}\r\n"
	"\t#endif\r\n"
	"\t\r\n"
	"\t#ifdef ENABLE_FOG_DEPTH\r\n"
	"\t\t// Mix the object's colour with the fogging colour based on fragment's depth\r\n"
	"\t\tlowp vec3 vFragColour = textureCube(CubeMap, EyeDir).rgb;\r\n"
	"\t\t\r\n"
	"\t\t// Test depth\r\n"
	"\t\tlowp float fFogBlend = clamp(VertexDepth * RcpMaxFogDepth, 0.0, 1.0);\r\n"
	"\t\tvFragColour.rgb = mix(vFragColour.rgb, FogColour.rgb, fFogBlend);\r\n"
	"\t\t\t\r\n"
	"\t\tgl_FragColor = vec4(vFragColour.rgb, 1.0);\r\n"
	"\t#else\r\n"
	"\t\tgl_FragColor = textureCube(CubeMap, EyeDir);\r\n"
	"\t\t\r\n"
	"\t#endif\r\n"
	"}";

// Register SkyboxFShader.fsh in memory file system at application startup time
static CPVRTMemoryFileSystem RegisterFile_SkyboxFShader_fsh("SkyboxFShader.fsh", _SkyboxFShader_fsh, 1035);

// ******** End: SkyboxFShader.fsh ********

// This file was created by Filewrap 1.1
// Little endian mode
// DO NOT EDIT

#include "../PVRTMemoryFileSystem.h"

// using 32 bit to guarantee alignment.
#ifndef A32BIT
 #define A32BIT static const unsigned int
#endif

// ******** Start: SkyboxFShader.fsc ********

// File data
A32BIT _SkyboxFShader_fsc[] = {
0x10fab438,0x1e87e07b,0x30050100,0x2101,0xa9142cc2,0x0,0x0,0x33020000,0x1000000,0x4000000,0x0,0x8000000,0x751803,0x1000100,0x20000,0x0,0x0,0xd8000000,0x55535020,0x17,0xcc,0x1,0x0,0x848,0x0,0x2,0x79,0x0,0x0,0x0,0xffffffff,0x0,0x770009,0xffff,0x1,0x0,0x1,0x0,0x0,0x0,0x0,0xfffc0000,0x3,0x10004,0x0,0x20000,0xffffffff,0x0,0x0,0x10000,0x0,0x4,0x50002,0x10000,0x80010000,0x80018001,0x8001,0x0,0x120000,0xf0000,0x60a000f,0x30003,0x30003,0x0,
0x0,0x40004,0x40004,0x10000,0x30002,0x1,0x30001,0x6,0xfa000000,0x55535020,0x17,0xee,0x1,0x0,0x948,0x0,0x2,0x79,0x0,0x0,0x0,0xffffffff,0x0,0x770009,0xffff,0x10001,0x0,0x1,0x0,0x0,0x0,0x0,0xfffc0000,0x1,0x10004,0x0,0x20000,0xffffffff,0x0,0x0,0x10000,0x0,0x4,0x50002,0x10000,0x80010000,0x80018001,0x8001,0x0,0x0,0xf0000,0x60a000f,0x30003,0x30003,0x0,0x0,0x40004,0x40004,0x10000,0x30002,0x10001,0x30001,0x2,0x1,
0x80018001,0x80018001,0x0,0x0,0x12,0x1000a000,0x62881,0x0,0x75430200,0x614d6562,0x70,0x31a,0x2000001,0x10000,0x45000001,0x69446579,0x72,0x504,0x1000001,0x30000,0x7,
};

// Register SkyboxFShader.fsc in memory file system at application startup time
static CPVRTMemoryFileSystem RegisterFile_SkyboxFShader_fsc("SkyboxFShader.fsc", _SkyboxFShader_fsc, 595);

// ******** End: SkyboxFShader.fsc ********

