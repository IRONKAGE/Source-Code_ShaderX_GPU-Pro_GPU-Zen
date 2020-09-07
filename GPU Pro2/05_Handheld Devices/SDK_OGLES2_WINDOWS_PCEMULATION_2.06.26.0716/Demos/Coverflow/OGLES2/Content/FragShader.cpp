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
	"varying mediump vec2   TexCoord;\r\n"
	"varying mediump vec4   Colors;\r\n"
	"\r\n"
	"void main()\r\n"
	"{\r\n"
	"    gl_FragColor = texture2D(sTexture, TexCoord) * Colors;\r\n"
	"}\r\n";

// Register FragShader.fsh in memory file system at application startup time
static CPVRTMemoryFileSystem RegisterFile_FragShader_fsh("FragShader.fsh", _FragShader_fsh, 179);

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
0x10fab438,0x7dde7163,0x30050100,0x2101,0xa9142cc2,0x0,0x0,0xac020000,0x1000000,0x4000000,0x0,0x18000000,0x750204,0x1000100,0x2020000,0x0,0x0,0x18010000,0x55535020,0x17,0x10c,0x1,0x0,0x848,0x0,0x2,0x79,0x0,0x0,0x0,0xffffffff,0x0,0x770009,0xffff,0x20003,0x0,0x2,0x0,0x0,0x0,0x0,0xfffc0000,0x20003,0x10004,0x0,0x20000,0xffffffff,0x0,0x60000,0x10000,0x20004,0x0,0x2,0x40000,0x20000,0x5,0x1,0x80018001,0x80018001,0x0,0x0,0x0,0xf000f,0x10688,
0x10001,0x1,0x10000,0x20001,0x20002,0x2,0x2,0x10002,0x20002,0x20002,0x30000,0x80010000,0x80018001,0x8001,0x0,0x0,0x100012,0x90000030,0x30a11002,0xa0448000,0x40801a3e,0xa0448081,0x40801a26,0x6,0x18010000,0x55535020,0x17,0x10c,0x1,0x0,0x948,0x0,0x2,0x79,0x0,0x0,0x0,0xffffffff,0x0,0x770009,0xffff,0x20003,0x0,0x2,0x0,0x0,0x0,0x0,0xfffc0000,0x1,0x10004,0x0,0x20000,0xffffffff,0x0,0x60000,0x10000,0x20004,0x0,0x2,0x40000,0x20000,0x5,0x1,
0x80018001,0x80018001,0x0,0x0,0x0,0xf000f,0x10688,0x10001,0x1,0x10000,0x20001,0x20002,0x2,0x2,0x10002,0x20002,0x20002,0x30000,0x80010000,0x80018001,0x8001,0x0,0x0,0x100012,0x90000030,0x30a11000,0x60000,0x40801a18,0x48081,0x40801a24,0x6,0x3000000,0x78655473,0x65727574,0x18000000,0x1000003,0x20000,0x1000100,0x65540000,0x6f6f4378,0x6472,0x50300,0x100,0x2040001,0x300,0x6f6c6f43,0x7372,0x50500,0x100,0x4000001,0xf00,0x0,
};

// Register FragShader.fsc in memory file system at application startup time
static CPVRTMemoryFileSystem RegisterFile_FragShader_fsc("FragShader.fsc", _FragShader_fsc, 716);

// ******** End: FragShader.fsc ********

