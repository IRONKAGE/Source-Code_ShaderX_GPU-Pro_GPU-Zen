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
	"uniform sampler2D\t\tNormalTex;\r\n"
	"uniform sampler2D\t\tReflectionTex;\r\n"
	"#ifdef ENABLE_REFRACTION\r\n"
	"\tuniform sampler2D\t\tRefractionTex;\r\n"
	"#endif\r\n"
	"uniform samplerCube NormalisationCubeMap;\r\n"
	"\r\n"
	"uniform mediump mat4\tModelViewMatrix;\r\n"
	"uniform lowp vec4\t\tWaterColour;\r\n"
	"#ifdef ENABLE_DISTORTION\r\n"
	"\tuniform mediump float\tWaveDistortion;\r\n"
	"#endif\r\n"
	"uniform mediump vec2 \tRcpWindowSize;\r\n"
	"\r\n"
	"varying mediump vec2 \tBumpCoord0;\r\n"
	"varying mediump vec2 \tBumpCoord1;\r\n"
	"varying mediump vec3\tWaterToEye;\r\n"
	"varying mediump float\tWaterToEyeLength;\r\n"
	"\r\n"
	"void main()\r\n"
	"{\t\r\n"
	"\t// Calculate the tex coords of the fragment (using it's position on the screen)\r\n"
	"\tlowp vec3 vAccumulatedNormal = vec3(0.0,1.0,0.0);\r\n"
	"\tmediump vec2 vTexCoord = gl_FragCoord.xy * RcpWindowSize;\r\n"
	"\t\r\n"
	"\t#ifdef ENABLE_DISTORTION\r\n"
	"\t\t// When distortion is enabled, use the normal map to calculate perturbation\r\n"
	"\t\tvAccumulatedNormal = texture2D(NormalTex, BumpCoord0).rgb;\r\n"
	"\t\tvAccumulatedNormal += texture2D(NormalTex, BumpCoord1).rgb;\r\n"
	"\t\tvAccumulatedNormal -= 1.0; // Same as * 2.0 - 2.0\r\n"
	"\t\r\n"
	"\t\tlowp vec2 vTmp = vAccumulatedNormal.xz;\r\n"
	"\t\t/* \t\r\n"
	"\t\t\tDivide by WaterToEyeLength to scale down the distortion\r\n"
	"\t\t \tof fragments based on their distance from the camera \r\n"
	"\t\t*/\r\n"
	"\t\tvTexCoord.xy -= vTmp * (WaveDistortion / length(WaterToEye));\r\n"
	"\t#endif\r\n"
	"\r\n"
	"#ifdef ENABLE_REFRACTION\r\n"
	"\tlowp vec4 vReflectionColour = texture2D(ReflectionTex, vTexCoord);\r\n"
	"\tlowp vec4 vRefractionColour = texture2D(RefractionTex, vTexCoord);\r\n"
	"\t\r\n"
	"\t#ifdef ENABLE_FRESNEL\r\n"
	"\t\t// Calculate the Fresnel term to determine amount of reflection for each fragment\r\n"
	"\t\t\r\n"
	"\t\t// Use normalisation cube map instead of normalize() - See section 3.3.1 of white paper for more info\r\n"
	"\t\tlowp vec3 vWaterToEyeCube = textureCube(NormalisationCubeMap,WaterToEye).rgb * 2.0 - 1.0;\r\n"
	"\r\n"
	"\t\tmediump float fEyeToNormalAngle = clamp(dot(vWaterToEyeCube, vAccumulatedNormal),0.0,1.0);\r\n"
	"\t\t\r\n"
	"\t\tmediump float fAirWaterFresnel = 1.0 - fEyeToNormalAngle;\r\n"
	"\t\tfAirWaterFresnel = pow(fAirWaterFresnel, 5.0);\r\n"
	"\t\tfAirWaterFresnel = (0.98 * fAirWaterFresnel) + 0.02;\t// R(0)-1 = ~0.98 , R(0)= ~0.02\r\n"
	"\t\tlowp float fTemp = fAirWaterFresnel;\r\n"
	"\t\t\r\n"
	"\t\t// Blend reflection and refraction\r\n"
	"\t\tgl_FragColor = mix(vRefractionColour, vReflectionColour, fTemp);\r\n"
	"\t#else\r\n"
	"\t\tgl_FragColor = mix(vRefractionColour, vReflectionColour, 0.4);\t// Constant mix\r\n"
	"\t#endif\r\n"
	"#else\r\n"
	"\tgl_FragColor = texture2D(ReflectionTex, vTexCoord);\t\t\t\t\t// Reflection only\r\n"
	"#endif\r\n"
	"}\r\n";

// Register FragShader.fsh in memory file system at application startup time
static CPVRTMemoryFileSystem RegisterFile_FragShader_fsh("FragShader.fsh", _FragShader_fsh, 2388);

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
0x10fab438,0x7ba31c5f,0x30050100,0x2101,0xa9142cc2,0x0,0x0,0x61030000,0x1000000,0x4000000,0x0,0x0,0x10000,0x0,0x100,0x0,0x1,0x56010000,0x55535020,0x17,0x14a,0x1,0x0,0x848,0x0,0x2,0x79,0x0,0x8,0x0,0xffffffff,0x0,0x770009,0xffff,0x10002,0x0,0x60001,0x0,0x0,0x0,0x0,0xfffc0000,0x3,0x10004,0x0,0x20000,0xffffffff,0x30000,0x40000,0xe0000,0x10002,0x40002,0x0,0x50002,0x100000,0xa0002,0x1,0xb0002,0x100001,0x80002,0x2,0x90002,0x100002,0x4,
0x20002,0x50000,0x80010000,0x80018001,0x8001,0x0,0x40000,0x40004,0x40004,0x70001a54,0x42100c,0x70005a55,0x42100e,0x70201a57,0x421086,0xd0002930,0x411002,0xd02069b0,0x411106,0x5,0x0,0x80018001,0x80018001,0x0,0x0,0x12,0xf000f,0x3060a,0x30003,0x3,0x0,0x40000,0x40004,0x4,0x20001,0x10003,0x1,0x1404a000,0x6e003,0x0,0x50204c01,0x175553,0x1400000,0x10000,0x0,0x9480000,0x0,0x20000,0x790000,0x0,0x80000,0x0,0xffff0000,0xffff,0x90000,0xffff0077,0x20000,0x3,0x10000,0x6,0x0,0x0,0x0,0x0,
0x1fffc,0x40000,0x1,0x0,0xffff0002,0xffff,0x3,0x4,0x2000e,0x20001,0x4,0x20000,0x5,0x20010,0x1000a,0x20000,0x1000b,0x20010,0x20008,0x20000,0x20009,0x40010,0x20000,0x2,0x4,0x80018001,0x80018001,0x0,0x0,0x40004,0x40004,0xf0202a52,0x401224,0x70005a55,0x42100c,0x70001a57,0x421080,0xd04029b0,0x411100,0x5,0x0,0x80018001,0x80018001,0x0,0x0,0x12,0xf000f,0x1060a,0x10001,0x1,0x0,0x40000,0x40004,0x4,0x20001,0x10003,0x40000003,0x1400a000,0x6e003,0xc000000,0x0,0x803f,0x0,0x0,
0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x63520300,0x6e695770,0x53776f64,0x657a69,0x3030000,0x10000,0x4000100,0x30002,0x66655200,0x7463656c,0x546e6f69,0x7865,0x31800,0x100,0x1000002,0x100,0x505f6c67,0x6f50584d,0x6a644173,0x747375,0x3055700,0x10000,0x8000100,0xf0004,0x0,
};

// Register FragShader.fsc in memory file system at application startup time
static CPVRTMemoryFileSystem RegisterFile_FragShader_fsc("FragShader.fsc", _FragShader_fsc, 897);

// ******** End: FragShader.fsc ********

