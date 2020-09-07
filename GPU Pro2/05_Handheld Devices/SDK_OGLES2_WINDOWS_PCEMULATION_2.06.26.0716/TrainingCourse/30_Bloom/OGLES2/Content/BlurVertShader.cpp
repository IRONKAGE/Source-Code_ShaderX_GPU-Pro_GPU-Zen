// This file was created by Filewrap 1.1
// Little endian mode
// DO NOT EDIT

#include "../PVRTMemoryFileSystem.h"

// using 32 bit to guarantee alignment.
#ifndef A32BIT
 #define A32BIT static const unsigned int
#endif

// ******** Start: BlurVertShader.vsh ********

// File data
static const char _BlurVertShader_vsh[] = 
	"// Blur filter kernel shader\r\n"
	"//\r\n"
	"// 0  1  2  3  4\r\n"
	"// x--x--X--x--x    <- original filter kernel\r\n"
	"//   y---X---y      <- filter kernel abusing the hardware texture filtering\r\n"
	"//       |\r\n"
	"//      texel center\r\n"
	"//\r\n"
	"// \r\n"
	"// Using hardware texture filtering, the amount of samples can be\r\n"
	"// reduced to three. To calculate the offset, use this formula:\r\n"
	"// d = w1 / (w1 + w2),  whereas w1 and w2 denote the filter kernel weights\r\n"
	"\r\n"
	"attribute highp   vec3  inVertex;\r\n"
	"attribute mediump vec2  inTexCoord;\r\n"
	"\r\n"
	"uniform mediump float  TexelOffsetX;\r\n"
	"uniform mediump float  TexelOffsetY;\r\n"
	"\r\n"
	"varying mediump vec2  TexCoord0;\r\n"
	"varying mediump vec2  TexCoord1;\r\n"
	"varying mediump vec2  TexCoord2;\r\n"
	"\r\n"
	"void main()\r\n"
	"{\r\n"
	"\t// Pass through vertex\r\n"
	"\tgl_Position = vec4(inVertex, 1.0);\r\n"
	"\t\r\n"
	"\t// Calculate texture offsets and pass through\t\r\n"
	"\tmediump vec2 offset = vec2(TexelOffsetX, TexelOffsetY);\r\n"
	"  \r\n"
	"    TexCoord0 = inTexCoord - offset;\r\n"
	"    TexCoord1 = inTexCoord;\r\n"
	"    TexCoord2 = inTexCoord + offset;    \r\n"
	"\r\n"
	"}";

// Register BlurVertShader.vsh in memory file system at application startup time
static CPVRTMemoryFileSystem RegisterFile_BlurVertShader_vsh("BlurVertShader.vsh", _BlurVertShader_vsh, 989);

// ******** End: BlurVertShader.vsh ********

// This file was created by Filewrap 1.1
// Little endian mode
// DO NOT EDIT

#include "../PVRTMemoryFileSystem.h"

// using 32 bit to guarantee alignment.
#ifndef A32BIT
 #define A32BIT static const unsigned int
#endif

// ******** Start: BlurVertShader.vsc ********

// File data
A32BIT _BlurVertShader_vsc[] = {
0x10fab438,0xc89286ce,0x30050100,0x2101,0xa9142cc2,0x0,0x0,0x44020000,0x0,0x4000000,0x0,0x39000000,0x20202,0x0,0x2020000,0x2,0x0,0x18010000,0x55535020,0x17,0x10c,0x1,0x0,0x80c,0x0,0x2,0x79,0x0,0x8,0x0,0xffffffff,0x0,0x76000a,0xffff,0x6,0x0,0x20000,0x0,0x0,0x0,0x0,0xfffc0000,0x0,0x0,0x0,0x20000,0xffffffff,0x10000,0x20006,0x4,0x20000,0x5,0x370010,0x40000,0x20000,0x2,0x10004,0x80018001,0x80018001,0x0,0x0,0x4,0x0,0x1,
0xfd100000,0xa0000000,0x28a12001,0x60601a00,0x28831001,0xa0c00200,0x28a11001,0x2,0x10004,0x80018004,0x80018001,0x0,0x0,0x4,0x10001,0x4010101,0xfa100000,0xf0812a54,0x38a30225,0xf1016ad5,0x38930625,0xe0a00000,0x28a31001,0x6,0x803f0800,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x8000000,0x505f6c67,0x7469736f,0x6e6f69,0x5050100,0x10000,0x100,0xf0004,0x566e6900,0x65747265,0x78,0x404,0x1000001,0x40000,0x54000007,0x6c657865,0x7366664f,0x587465,0x3020000,0x10000,0x4000100,0x10001,0x78655400,0x664f6c65,0x74657366,0x59,0x302,0x1000001,0x10500,0x54000001,0x6f437865,0x3064726f,
0x3000000,0x1000005,0x10000,0x3000200,0x6e690000,0x43786554,0x64726f6f,0x3000000,0x1000004,0x10000,0x3000404,0x65540000,0x6f6f4378,0x316472,0x5030000,0x10000,0x4000100,0x30002,0x78655400,0x726f6f43,0x3264,0x50300,0x100,0x2080001,0x300,0x0,
};

// Register BlurVertShader.vsc in memory file system at application startup time
static CPVRTMemoryFileSystem RegisterFile_BlurVertShader_vsc("BlurVertShader.vsc", _BlurVertShader_vsc, 612);

// ******** End: BlurVertShader.vsc ********

