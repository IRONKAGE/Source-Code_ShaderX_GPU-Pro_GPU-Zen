// This file was created by Filewrap 1.1
// Little endian mode
// DO NOT EDIT

#include "../PVRTMemoryFileSystem.h"

// using 32 bit to guarantee alignment.
#ifndef A32BIT
 #define A32BIT static const unsigned int
#endif

// ******** Start: VertShader.vsh ********

// File data
static const char _VertShader_vsh[] = 
	"/******************************************************************************\r\n"
	"* Vertex Shader\r\n"
	"******************************************************************************/\r\n"
	"/* \r\n"
	"\tThe vertex and fragment shaders implement two techniques for reflections.\r\n"
	"\tWhich version is used for drawing the object is dependent on the value of \r\n"
	"\tbHighDetail. If bHighDetail is true it uses the method described in \r\n"
	"\tOGLES2PerturbedUVs and for false it uses OGLES2Reflections. \r\n"
	"\t\r\n"
	"\tReason for using 2 methods is that when the object is far away you aren't\r\n"
	"\tgoing to notice all the detail that the PerturbedUVs method adds to\r\n"
	"\tthe mesh so you may as well use a simpler method for creating reflections.\r\n"
	"\tThis way you aren't using valuable resources on something you aren't \r\n"
	"\tgoing to notice.\r\n"
	"\r\n"
	"\tAlso, when the training course is in 'low detail' mode it uses a different mesh.\r\n"
	"\tThe mesh that is being drawn contains only 7% of the original meshes vertices.\r\n"
	"*/\r\n"
	"\r\n"
	"attribute highp   vec3  inVertex;\r\n"
	"attribute mediump vec3  inNormal;\r\n"
	"attribute mediump vec2  inTexCoord;\r\n"
	"attribute mediump vec3  inTangent;\r\n"
	"\r\n"
	"uniform highp   mat4  MVPMatrix;\r\n"
	"uniform mediump mat3  ModelWorld;\r\n"
	"uniform mediump vec3  EyePosModel;\r\n"
	"uniform bool          bHighDetail;\r\n"
	"\r\n"
	"varying mediump vec3  EyeDirection;\r\n"
	"varying mediump vec2  TexCoord;\r\n"
	"\r\n"
	"void main()\r\n"
	"{\r\n"
	"\t// Transform position\r\n"
	"\tgl_Position = MVPMatrix * vec4(inVertex,1.0);\r\n"
	"\r\n"
	"\t// Calculate direction from eye position in model space\r\n"
	"\tmediump vec3 eyeDirModel = normalize(EyePosModel - inVertex);\r\n"
	"\r\n"
	"\tif (bHighDetail)\r\n"
	"\t{\t\r\n"
	"\t\t// transform light direction from model space to tangent space\r\n"
	"\t\tmediump vec3 binormal = cross(inNormal, inTangent);\r\n"
	"\t\tmediump mat3 tangentSpaceXform = mat3(inTangent, binormal, inNormal);\r\n"
	"\t\tEyeDirection = eyeDirModel * tangentSpaceXform;\t\r\n"
	"\r\n"
	"\t\tTexCoord = inTexCoord;\r\n"
	"\t}\r\n"
	"\telse\r\n"
	"\t{\r\n"
	"\t\t// reflect eye direction over normal and transform to world space\r\n"
	"\t\tmediump vec3 reflectDir = ModelWorld * reflect(eyeDirModel, inNormal);\r\n"
	"\t\tTexCoord = normalize(reflectDir).xy * 0.5 + 0.5;\r\n"
	"\t}\r\n"
	"}\r\n";

// Register VertShader.vsh in memory file system at application startup time
static CPVRTMemoryFileSystem RegisterFile_VertShader_vsh("VertShader.vsh", _VertShader_vsh, 2049);

// ******** End: VertShader.vsh ********

// This file was created by Filewrap 1.1
// Little endian mode
// DO NOT EDIT

#include "../PVRTMemoryFileSystem.h"

// using 32 bit to guarantee alignment.
#ifndef A32BIT
 #define A32BIT static const unsigned int
#endif

// ******** Start: VertShader.vsc ********

// File data
A32BIT _VertShader_vsc[] = {
0x10fab438,0xcaf61986,0x30050100,0x2101,0xa9142cc2,0x0,0x0,0x29070000,0x0,0x4000000,0x0,0x19000000,0x203,0x1000000,0x2020000,0x0,0x200,0x2c050000,0x55535020,0x17,0x520,0x1,0x0,0x80c,0x0,0x2,0x79,0x0,0x8,0x0,0xffffffff,0x0,0x76000a,0xffff,0x7000e,0x0,0x250000,0x0,0x0,0x0,0x0,0xfffc0000,0x0,0x0,0x0,0x20000,0xffffffff,0x1b0000,0x2000e,0x1a,0x20000,0x1b,0x20010,0x10014,0x20000,0x10015,0x20010,0x2001c,0x20000,0x2001d,0x20010,0x30020,0x20000,0x30021,
0x20010,0x40024,0x20000,0x40025,0x10010,0x50004,0x10000,0x60005,0x10000,0x70006,0x10000,0x80007,0x10000,0x90008,0x10000,0xa0009,0x10000,0xb000a,0x10000,0xc000b,0x10000,0xd000c,0x10000,0xe000d,0x10000,0xf000e,0x10000,0x10000f,0x10000,0x110010,0x10000,0x120011,0x10000,0x130012,0x10000,0x140013,0x20000,0x150016,0x20000,0x150017,0x10010,0x160017,0x20000,0x170018,0x20000,0x170019,0x20010,0x18001e,0x20000,0x18001f,0x20010,0x190022,0x20000,0x190023,0x20010,0x1a0026,0x20000,0x1a0027,0x37770010,0x40000,0x20000,0x2,0x10011,0x80018001,
0x80018001,0x0,0x0,0x4,0x0,0x0,0x0,0x0,0x10001,0x1,0x10001,0x10001,0x0,0x1a56fd10,0x10847040,0x5a5700c2,0x10847060,0x9a7e00c2,0x10847000,0x818300c2,0x14020060,0x30080,0x10022060,0x1800080,0x12008020,0xb00880,0x10001080,0x10300081,0x1281d000,0x7904889,0x606f060,0x499438ab,0x606f060,0x8b983882,0x606f060,0x1a1b3882,0x1001701f,0x89200a2,0x606f000,0x4a9638ab,0x606f000,0x8c9a3882,0x606f000,0x1a1d3882,0x1001705f,0x200a2,0x20000,0x80010001,0x80018000,0x8000,0x0,0x40000,0x1000000,0x100,0x4130fa10,0x100010a0,0x300a1,0x0,0x400000,0x3fd00,0x2,0x1000d,0x80008001,0x80008001,
0x0,0x0,0x0,0x10001,0x10001,0x0,0x10001,0x10001,0x6000001,0x1001a0e0,0x50528a1,0x6068001,0x840038ab,0x100eb041,0x82810081,0x1003b001,0x82090089,0x20520c1,0x450138aa,0x100eb001,0x40300081,0x10029021,0x2850081,0x6068002,0x48038ab,0x100eb001,0x84810081,0x1003b001,0x81010089,0x1003a021,0x5000088,0x201a081,0x8038a2,0x1001e0a0,0x32883,0x0,0x400000,0x4f800,0x4,0x20003,0x20000,0x80000000,0x80018001,0x4089,0x0,0x40000,0x1890000,0x10001,0x204fa10,0x20028000,0x210a0,0x10000,0x80000000,0x80018001,0x4089,0x430000,0x40000,0x43000,0xfb100000,0x2,0x2,0x80018001,0x80018001,0x0,
0x43,0x4,0x1010101,0xfa100000,0xd0002030,0xc11006,0x2,0x1,0x80018001,0x80018001,0x0,0x0,0x4,0x0,0x2fb10,0x20000,0x80010000,0x80018000,0x4089,0x0,0x40000,0x1890000,0x10100,0x204fa10,0x20868020,0x200a0,0x10000,0x80010000,0x80018000,0x4089,0x440000,0x40000,0x44000,0xfb100000,0x2,0x2,0x80028000,0x80018002,0x0,0x44,0x4,0x20201,0xfa100000,0xe0802001,0x10e02002,0x2,0x1,0x80018001,0x80008000,0x0,0x44,0x4,0x101,0x2fa10,0x90000,0x80010000,0x80008001,0x8000,0x0,0x40000,0x10000,0x10001,0x10001,0x10001,
0x0,0xfb100000,0xf0006c59,0x38ab0606,0xf000ad5b,0x38820606,0xf000ee5d,0x38820606,0xf01f0081,0x831402,0xa0010200,0x801006,0x80000000,0x8801202,0xa0000000,0x3888c606,0xf0ef2a54,0xe01001,0x4,0x20004,0x10000,0x80010000,0x80018001,0x8001,0x0,0x40000,0x1010101,0xfa100000,0x6,0x803f2800,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x400000,0x3f0000,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,
0x0,0x0,0x0,0x0,0x0,0xb000000,0x65566e69,0x78657472,0x4000000,0x1000004,0x10000,0x7000400,0x6c670000,0x736f505f,0x6f697469,0x100006e,0x505,0x1000001,0x40000,0x4d00000f,0x614d5056,0x78697274,0x16000000,0x1000003,0x10000,0xffff1004,0x79450000,0x736f5065,0x65646f4d,0x6c,0x304,0x1000001,0x31400,0x62000007,0x68676948,0x61746544,0x6c69,0x30a00,0x100,0x1170001,0x100,0x6f4e6e69,0x6c616d72,0x4000000,0x1000004,0x10000,0x7000404,0x6e690000,0x676e6154,0x746e65,0x4040000,0x10000,0x8000100,0x70004,0x65794500,0x65726944,0x6f697463,0x6e,0x504,0x1000001,0x30000,0x54000007,0x6f437865,0x64726f,
0x5030000,0x10000,0x4000100,0x30002,0x546e6900,0x6f437865,0x64726f,0x4030000,0x10000,0xc000100,0x30004,0x646f4d00,0x6f576c65,0x646c72,0x3120000,0x10000,0x1c000100,0x77070c,0x0,
};

// Register VertShader.vsc in memory file system at application startup time
static CPVRTMemoryFileSystem RegisterFile_VertShader_vsc("VertShader.vsc", _VertShader_vsc, 1865);

// ******** End: VertShader.vsc ********

