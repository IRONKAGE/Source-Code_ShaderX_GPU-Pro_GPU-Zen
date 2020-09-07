// This file was created by Filewrap 1.1
// Little endian mode
// DO NOT EDIT

#include "../PVRTMemoryFileSystem.h"

// using 32 bit to guarantee alignment.
#ifndef A32BIT
 #define A32BIT static const unsigned int
#endif

// ******** Start: FragShader.fsh ********

// File data
static const char _FragShader_fsh[] = 
	"uniform sampler2D  sTexture;\r\n"
	"\r\n"
	"varying mediump vec2  TexCoord;\r\n"
	"\r\n"
	"void main()\r\n"
	"{\r\n"
	"    gl_FragColor = texture2D(sTexture, TexCoord);\r\n"
	"}\r\n";

// Register FragShader.fsh in memory file system at application startup time
static CPVRTMemoryFileSystem RegisterFile_FragShader_fsh("FragShader.fsh", _FragShader_fsh, 137);

// ******** End: FragShader.fsh ********

// This file was created by Filewrap 1.1
// Little endian mode
// DO NOT EDIT

#include "../PVRTMemoryFileSystem.h"

// using 32 bit to guarantee alignment.
#ifndef A32BIT
 #define A32BIT static const unsigned int
#endif

// ******** Start: FragShader.fsc ********

// File data
A32BIT _FragShader_fsc[] = {
0x10fab438,0x3dd66b44,0x30050100,0x2101,0xa9142cc2,0x0,0x0,0x36020000,0x1000000,0x4000000,0x0,0x8000000,0x4402,0x0,0x20000,0x0,0x0,0xd8000000,0x55535020,0x17,0xcc,0x1,0x0,0x848,0x0,0x2,0x79,0x0,0x0,0x0,0xffffffff,0x0,0x770009,0xffff,0x1,0x0,0x1,0x0,0x0,0x0,0x0,0xfffc0000,0x3,0x10004,0x0,0x20000,0xffffffff,0x0,0x0,0x10000,0x0,0x4,0x50002,0x10000,0x80010000,0x80018001,0x8001,0x0,0x120000,0xf0000,0x60a000f,0x30003,0x30003,0x0,
0x0,0x40004,0x40004,0x10000,0x30002,0x1,0x20001,0x6,0xfa000000,0x55535020,0x17,0xee,0x1,0x0,0x948,0x0,0x2,0x79,0x0,0x0,0x0,0xffffffff,0x0,0x770009,0xffff,0x10001,0x0,0x1,0x0,0x0,0x0,0x0,0xfffc0000,0x1,0x10004,0x0,0x20000,0xffffffff,0x0,0x0,0x10000,0x0,0x4,0x50002,0x10000,0x80010000,0x80018001,0x8001,0x0,0x0,0xf0000,0x60a000f,0x30003,0x30003,0x0,0x0,0x40004,0x40004,0x10000,0x30002,0x10001,0x20001,0x2,0x1,
0x80018001,0x80018001,0x0,0x0,0x12,0x1000a000,0x62881,0x0,0x54730200,0x75747865,0x6572,0x31800,0x100,0x1000002,0x100,0x43786554,0x64726f6f,0x3000000,0x1000005,0x10000,0x3000200,0x0,
};

// Register FragShader.fsc in memory file system at application startup time
static CPVRTMemoryFileSystem RegisterFile_FragShader_fsc("FragShader.fsc", _FragShader_fsc, 598);

// ******** End: FragShader.fsc ********

