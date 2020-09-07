// This file was created by Filewrap 1.1
// Little endian mode
// DO NOT EDIT

#include "../PVRTMemoryFileSystem.h"

// using 32 bit to guarantee alignment.
#ifndef A32BIT
 #define A32BIT static const unsigned int
#endif

// ******** Start: PostBloomVertShader.vsh ********

// File data
static const char _PostBloomVertShader_vsh[] = 
	"attribute highp   vec2  inVertex;\r\n"
	"attribute mediump vec2  inTexCoord;\r\n"
	"\r\n"
	"varying mediump vec2   TexCoord;\r\n"
	"\r\n"
	"void main()\r\n"
	"{\r\n"
	"    // Pass through vertex\r\n"
	"\tgl_Position = vec4(inVertex, 0.0, 1.0);\r\n"
	"\t\r\n"
	"\t// Pass through texcoords\r\n"
	"\tTexCoord = inTexCoord;\r\n"
	"}\r\n";

// Register PostBloomVertShader.vsh in memory file system at application startup time
static CPVRTMemoryFileSystem RegisterFile_PostBloomVertShader_vsh("PostBloomVertShader.vsh", _PostBloomVertShader_vsh, 255);

// ******** End: PostBloomVertShader.vsh ********

// This file was created by Filewrap 1.1
// Little endian mode
// DO NOT EDIT

#include "../PVRTMemoryFileSystem.h"

// using 32 bit to guarantee alignment.
#ifndef A32BIT
 #define A32BIT static const unsigned int
#endif

// ******** Start: PostBloomVertShader.vsc ********

// File data
A32BIT _PostBloomVertShader_vsc[] = {
0x10fab438,0xf049529c,0x30050100,0x2101,0xa9142cc2,0x0,0x0,0xab010000,0x0,0x4000000,0x0,0x9000000,0x2,0x0,0x20000,0x0,0x0,0x2010000,0x55535020,0x17,0xf6,0x1,0x0,0x80c,0x0,0x2,0x79,0x0,0x0,0x0,0xffffffff,0x0,0x76000a,0xffff,0x6,0x0,0x0,0x0,0x0,0x0,0x0,0xfffc0000,0x0,0x0,0x0,0x20000,0xffffffff,0x0,0x330006,0x40000,0x20000,0x2,0x1,0x80018001,0x80018001,0x0,0x0,0x0,0x1001a000,0x228a1,0x20000,0x80010000,0x80048004,0x8001,
0x0,0x40000,0x4010000,0x104,0x1800fa10,0x10016040,0x228a3,0x20000,0x80010000,0x80018001,0x8001,0x0,0x40000,0x1010000,0x101,0x200fa10,0x1001a080,0x628a1,0x4000000,0x0,0x803f,0x0,0x0,0x6c670400,0x736f505f,0x6f697469,0x100006e,0x505,0x1000001,0x40000,0x6900000f,0x7265566e,0x786574,0x4030000,0x10000,0x100,0x30004,0x78655400,0x726f6f43,0x64,0x503,0x1000001,0x20000,0x69000003,0x7865546e,0x726f6f43,0x64,0x403,0x1000001,0x40400,0x3,
};

// Register PostBloomVertShader.vsc in memory file system at application startup time
static CPVRTMemoryFileSystem RegisterFile_PostBloomVertShader_vsc("PostBloomVertShader.vsc", _PostBloomVertShader_vsc, 459);

// ******** End: PostBloomVertShader.vsc ********

