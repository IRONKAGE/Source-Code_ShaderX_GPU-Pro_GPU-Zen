#!BPY

"""
Name: 'POD (.pod)'
Blender: 248
Group: 'Export'
Tooltip: 'exporter to POD file format'
"""

bppversion = "0.1"	# version of library(so,dll) to load

import sys

from ctypes import *
from Blender import *
from os import *
from math import *
from operator import itemgetter

# Some Constants and defines

MAXIMAL_VARIATION = 0.00001

EPODDataNone = 0
EPODDataFloat = 1
EPODDataInt = 2
EPODDataUnsignedShort = 3
EPODDataRGBA = 4
EPODDataARGB = 5
EPODDataD3DCOLOR = 6
EPODDataUBYTE4 = 7
EPODDataDEC3N = 8
EPODDataFixed16_16 = 9
EPODDataUnsignedByte = 10
EPODDataShort = 11
EPODDataShortNorm = 12
EPODDataByte = 13
EPODDataByteNorm = 14

types_dict = { EPODDataNone : "EPODDataNone",
EPODDataFloat : "float",
EPODDataInt : "int",
EPODDataUnsignedShort : "unsigned short",
EPODDataRGBA : "RGBA",
EPODDataARGB : "ARGB",
EPODDataD3DCOLOR : "D3DCOLOR",
EPODDataUBYTE4 : "EPODDataUBYTE4",	# this one does not correspond to anything 
EPODDataDEC3N : "DEC3N",
EPODDataFixed16_16 : "fixed 16.16",
EPODDataUnsignedByte : "unsigned byte",
EPODDataShort : "short",
EPODDataShortNorm : "short, normalised",
EPODDataByte : "byte",
EPODDataByteNorm : "byte, normalised"
}

types_indices = { "ARGB" : 1,
				  "byte":2,
				  "byte, normalised":3,
				  "D3DCOLOR":4,
				  "DEC3N":5,
				  "fixed 16.16":6,
				  "float":7,
				  "int":8,
				  "RGBA":9,
				  "short":10,
				  "short, normalised":11,
				  "unsigned byte":12,
				  "unsigned short":13}


# ESkinType
ePODSkin = 0
ePhysiqueSkin = 1
# EPODLight
ePODPoint=0		 	# Point light 
ePODDirectional=1  # Directional light 
ePODSpot=2		 	#  Spot light 
# EPODCamera
eFreeCamera = 0
eTargetCamera = 1

#ECNodeTypes
eDummyCNode = 0
eMeshCNode = 1
eCameraCNode = 2
eLightCNode = 3
eBoneCNode = 4


DLGOPT_UVW_OPT_REMEMBERED = 8
DLGOPT_VECTOR_X = 0x01
DLGOPT_VECTOR_Y	= 0x02
DLGOPT_VECTOR_Z	= 0x04
DLGOPT_VECTOR_W	= 0x08
DLGOPT_VECTOR_MASK = 0x0F

MAX_UV_COORD = 8

eOGL = 0
eD3D = 1

ePOD = 0
eH = 1

eNone = 0
e590Blocks = 1
eD3DX = 2
ePVRTTriStrip = 3

eList = 0
eStrip = 1

PODfilename = "obiekt.pod"

# ID that should be unique for each node
uniqueID = 0


# Events identifiers list
ID_POSITION_EVT_TYPE = 1110
ID_POSITION_EVT_A1 = 1111
ID_POSITION_EVT_A2 = 1112
ID_POSITION_EVT_A3 = 1113
ID_NORMAL_EVT_TYPE = 1120
ID_NORMAL_EVT_A1 = 1121
ID_NORMAL_EVT_A2 = 1122
ID_NORMAL_EVT_A3 = 1123
ID_TANGENT_EVT_TYPE = 1130
ID_TANGENT_EVT_A1 = 1131
ID_TANGENT_EVT_A2 = 1132
ID_TANGENT_EVT_A3 = 1133
ID_BINORMAL_EVT_TYPE = 1140
ID_BINORMAL_EVT_A1 = 1141
ID_BINORMAL_EVT_A2 = 1142
ID_BINORMAL_EVT_A3 = 1143
ID_COLOUR_EVT_TYPE = 1260
ID_COLOUR_EVT_A1 = 1261
ID_COLOUR_EVT_A2 = 1262
ID_COLOUR_EVT_A3 = 1263
ID_COLOUR_EVT_A4 = 1264
ID_BONE_INDICES_EVT_TYPE = 1150
ID_BONE_INDICES_EVT_A1 = 1151
ID_BONE_INDICES_EVT_A2 = 1152
ID_BONE_INDICES_EVT_A3 = 1153
ID_BONE_INDICES_EVT_A4 = 1154
ID_BONE_WEIGHTS_EVT_TYPE = 1160	
ID_BONE_WEIGHTS_EVT_A1 = 1161	
ID_BONE_WEIGHTS_EVT_A2 = 1162
ID_BONE_WEIGHTS_EVT_A3 = 1163
ID_BONE_WEIGHTS_EVT_A4 = 1164
ID_UVW0_EVT_TYPE = 1170
ID_UVW0_EVT_A1 = 1171
ID_UVW0_EVT_A2 = 1172
ID_UVW0_EVT_A3 = 1173
ID_UVW1_EVT_TYPE = 1180
ID_UVW1_EVT_A1 = 1181
ID_UVW1_EVT_A2 = 1182
ID_UVW1_EVT_A3 = 1183
ID_UVW2_EVT_TYPE = 1190
ID_UVW2_EVT_A1 = 1191
ID_UVW2_EVT_A2 = 1192
ID_UVW2_EVT_A3 = 1193
ID_UVW3_EVT_TYPE = 1210
ID_UVW3_EVT_A1 = 1211
ID_UVW3_EVT_A2 = 1212
ID_UVW3_EVT_A3 = 1213
ID_UVW4_EVT_TYPE = 1220
ID_UVW4_EVT_A1 = 1221
ID_UVW4_EVT_A2 = 1222
ID_UVW4_EVT_A3 = 1223
ID_UVW5_EVT_TYPE = 1230
ID_UVW5_EVT_A1 = 1231
ID_UVW5_EVT_A2 = 1232
ID_UVW5_EVT_A3 = 1233
ID_UVW6_EVT_TYPE = 1240
ID_UVW6_EVT_A1 = 1241
ID_UVW6_EVT_A2 = 1242
ID_UVW6_EVT_A3 = 1243
ID_UVW7_EVT_TYPE = 1250
ID_UVW7_EVT_A1 = 1251
ID_UVW7_EVT_A2 = 1252
ID_UVW7_EVT_A3 = 1253

primitive_type_menu = 0
triangle_sorting_method_menu = 0
static_frame_button = 0
max_simulatnous_matrices_button = 0
position_data_type_menu = 0
normal_data_type_menu = 0
tangent_data_type_menu = 0
binormal_data_type_menu = 0
colour_data_type_menu = 0
bone_indices_data_type_menu = 0
bone_weights_data_type_menu = 0
uvw0_data_type_menu = 0
uvw1_data_type_menu = 0
uvw2_data_type_menu = 0
uvw3_data_type_menu = 0
uvw4_data_type_menu = 0
uvw5_data_type_menu = 0
uvw6_data_type_menu = 0
uvw7_data_type_menu = 0

listOfAddedObjects = {}


class MyVectorOpt(Structure): 
	_fields_ = [("eType",c_int),("nEnable",c_uint)]	
	def __init__(self):
		self.eType = EPODDataFloat 
		self.nEnable = DLGOPT_VECTOR_X | DLGOPT_VECTOR_Y | DLGOPT_VECTOR_Z | DLGOPT_VECTOR_W | DLGOPT_VECTOR_MASK

class FormatDescription:
	def __init__(self):
		self.type = 0
		self.evt_type = 0
		self.evt1 = 0
		self.evt2 = 0
		self.evt3 = 0
		self.evt4 = 0
		self.enabled = 0
	def __init__(self,atype,aevt_type,aevt1,aevt2,aevt3,aevt4,aenabled):		
		self.type = atype
		self.evt_type = aevt_type
		self.evt1 = aevt1
		self.evt2 = aevt2
		self.evt3 = aevt3
		self.evt4 = aevt4
		self.enabled = aenabled
		
class MyOptions(Structure):
	_fields_ = [("staticFrame",c_int),("eExpFormat",c_int),("exportGeom",c_int),("exportNormals",c_int),("exportVertexColor",c_int),("exportControllers",c_int),
				("exportMappingChannel",c_int),("exportMaterials",c_int),("exportSplines",c_int),("exportSkin",c_int),("exportObjectSpace",c_int),
				("exportBoneGeometry",c_int),("exportMatrices",c_int),("cS",c_int),("dwBoneLimit",c_uint),("eTriSort",c_int),("bSortVtx",c_int),("ePrimType",c_int),
				("bIndexed",c_int),("bInterleaved",c_int),("bTangentSpace",c_int),("bFixedPoint",c_int),("bFlipTextureV",c_int),("fTangentSpaceVtxSplit",c_float),
				("sVcOptPos",MyVectorOpt),("sVcOptNor",MyVectorOpt),("sVcOptTan",MyVectorOpt),("sVcOptBin",MyVectorOpt),("sVcOptCol",MyVectorOpt),
				("sVcOptBoneInd",MyVectorOpt),("sVcOptBoneWt",MyVectorOpt),("psVcOptUVW", DLGOPT_UVW_OPT_REMEMBERED * MyVectorOpt),
				("sClPostTarget",c_char_p),("sClPostArgs",c_char_p),("sClPostCWD",c_char_p),("m_sSerialisedOpt",c_char_p)]
	def __init__(self):
		# set default values of export options			
		self.staticFrame = 0
		self.eExpFormat = ePOD
		self.exportGeom = 1
		self.exportNormals = 1
		self.exportVertexColor = 0
		self.exportControllers = 0
		self.exportMappingChannel = 1
		self.exportMaterials = 1
		self.exportSplines = 0
		self.exportSkin = 0
		self.exportObjectSpace = 1
		self.exportBoneGeometry = 0
		self.exportMatrices = 0
		self.cS = eOGL
		self.dwBoneLimit = 9	
		self.eTriSort = eNone
		self.bSortVtx = 0
		self.ePrimType = eList
		self.bIndexed = 1
		self.bInterleaved = 1
		self.bTangentSpace = 0
		self.bFixedPoint = 0
		self.bFlipTextureV = 0
		self.fTangentSpaceVtxSplit	= 0
		self.sVcOptPos = MyVectorOpt()
		self.sVcOptNor = MyVectorOpt()
		self.sVcOptTan = MyVectorOpt()
		self.sVcOptBin = MyVectorOpt()
		self.sVcOptCol = MyVectorOpt()
		self.sVcOptCol.eType		= EPODDataARGB
		self.sVcOptCol.nEnable		= DLGOPT_VECTOR_X | DLGOPT_VECTOR_Y | DLGOPT_VECTOR_Z | DLGOPT_VECTOR_W
		self.sVcOptBoneInd.eType	= EPODDataUnsignedByte
		self.sVcOptBoneInd.nEnable	= DLGOPT_VECTOR_X | DLGOPT_VECTOR_Y | DLGOPT_VECTOR_Z | DLGOPT_VECTOR_W
		self.sVcOptBoneWt.eType	= EPODDataFloat
		self.sVcOptBoneWt.nEnable	= DLGOPT_VECTOR_X | DLGOPT_VECTOR_Y | DLGOPT_VECTOR_Z | DLGOPT_VECTOR_W
	
		for i in range(0,DLGOPT_UVW_OPT_REMEMBERED): 
			self.psVcOptUVW[i] =  MyVectorOpt()
			self.psVcOptUVW[i].nEnable &= ~DLGOPT_VECTOR_Z
	
		self.sClPostTarget = ""
		self.sClPostArgs = ""
		self.sClPostCWD = ""

# Export options
export_options = MyOptions()

# additional globals
chosen_page = 0
number_of_raw = 0


class MyVec3f(Structure):
	_fields_ = [("x",c_float),("y",c_float),("z",c_float)]
	def __init__(self,ax,ay,az):
		self.x = c_float(ax)
		self.y = c_float(ay)
		self.z = c_float(az)
	

class MyVec4f(Structure):
	_fields_ = [("x",c_float),("y",c_float),("z",c_float), ("w",c_float)]

class MyQuaternionf(Structure):
	_fields_ = [("x",c_float),("y",c_float),("z",c_float),("w",c_float)]

class MyMatrixf(Structure):
	_fields_ = [("f",c_float * 16)]
	
class SceneInfo(Structure):
	_fields_ = [("m_startFrame",c_int),("m_endFrame",c_int),("m_ambient",MyVec3f)]

class MyMaterial(Structure):
	_fields_ = [("sName",c_char_p),("sTexDiffuse",c_char_p),("sTexAmbient",c_char_p),("sTexSpecularColour",c_char_p),
				("sTexSpecularLevel",c_char_p),("sTexBump",c_char_p),("sTexEmissive",c_char_p),
				("sTexGlossiness",c_char_p),("sTexOpacity",c_char_p),("sTexReflection",c_char_p),
				("sTexRefraction",c_char_p),("vDiffuse",MyVec3f),("vAmbient",MyVec3f),("vSpecular",MyVec3f),
				("fOpacity",c_float),("fGlossiness",c_float),("fSpecularLevel",c_float),
				("sPFXFileName",c_char_p),("sPFXEffectName",c_char_p)]
	def __init__(self):
		self.sName = c_char_p("")
		self.fOpacity = c_float(0.0)
		self.fGlossiness = c_float(0.0)
		self.fSpecularLevel = c_float(0.0)
		self.vDiffuse.x = self.vDiffuse.y = self.vDiffuse.z = c_float(0.0)
		self.vAmbient.x = self.vAmbient.y = self.vAmbient.z = c_float(0.0)
		self.vSpecular.xyz = c_float(0.0)
		self.sTexDiffuse = c_char_p("")
		self.sTexAmbient = c_char_p("")
		self.sTexSpecularColour = c_char_p("")
		self.sTexSpecularLevel = c_char_p("")
		self.sTexBump = c_char_p("")
		self.sTexEmissive = c_char_p("")
		self.sTexGlossiness = c_char_p("")
		self.sTexOpacity = c_char_p("")
		self.sTexReflection = c_char_p("")
		self.sTexRefraction = c_char_p("")
		self.sPFXFileName = c_char_p("")
		self.sPFXEffectName = c_char_p("")

class SScale(Structure): 
	_fields_ = [ ("f0",c_float),("f1",c_float) ,("f2",c_float) ,("f3",c_float) ,("f4",c_float) , ("f5",c_float), ("f6",c_float) ]

class SLightData(Structure): 
	_fields_ = [("eType",c_int),("vColour",MyVec3f),("nTargetID",c_int),("fConstantAttenuation",c_float),("fLinearAttenuation",c_float),
	("fQuadraticAttenuation",c_float),("fFalloffAngle",c_float),("fFalloffExponent",c_float)]
	def __init__(self):
		self.eType = 0
		self.nTargetID = -1
		self.fConstantAttenuation  = c_float(1.0)
		self.fLinearAttenuation    = c_float(0.0)
		self.fQuadraticAttenuation = c_float(0.0)
		self.fFalloffAngle		   = c_float(radians(180))
		self.fFalloffExponent	   = c_float(0.0)


class SCameraData(Structure):
	_fields_ = [("eType",c_int),("fFOV",c_float),("fFarClip",c_float),("fNearClip",c_float),("nTargetID",c_int)]	
	def __init__(self):
		self.eType = 0
		self.nTargetID = -1
		self.fFOV = 0.0

class SNodeTM(Structure):
	_fields_ = [("vPos",MyVec3f),("qRot",MyQuaternionf),("sScale",SScale),("fMatrix",MyMatrixf)]

class CBoneWeight(Structure):
	_fields_ = [("nBone",c_int),("fWeight",c_float)]
	def __init__(self):
		self.nBone = -1
		self.fWeight = 1.0

class SBonedVtx(Structure): 
	_fields_ = [("vBone",POINTER(CBoneWeight)),("numvBone",c_int)]
	def __init__(self):
		self.vBone = None
		self.numvBone = 0

class SModifier(Structure):
	_fields_ = [("sName",c_char_p),("eSkinType",c_int),("bIsSkin",c_int),("vVtx",POINTER(SBonedVtx)),("numvVtx",c_int)] 	
	def __init__(self):
		self.sName = c_char_p("")
		self.eSkinType = ePODSkin
		self.bIsSkin = 0 # False - no skinning at the beginning
		self.numvVtx = 0
		self.vVtx = None

class CTriIdx(Structure):
	_fields_ = [("i",c_int * 3)]
	def __init__(self):
		self.i[0] = -1
		self.i[1] = -1
		self.i[2] = -1

class SFace(Structure): 
	_fields_ = [("sVtx",CTriIdx),("smGrp",c_int),("sNrm",CTriIdx),("sVtxCol",CTriIdx)]	
	def __init__(self):
		self.sVtx = CTriIdx()
		self.sNrm = CTriIdx()
		self.sVtxCol = CTriIdx()	

class SMappingChannel(Structure):  
	_fields_ = [("vVtx",POINTER(MyVec3f)),("vIdx",POINTER(CTriIdx)),("numvVtx",c_int),("numvIdx",c_int) ]		
	def __init__(self):
		self.vVtx = None
		self.numvVtx = 0
		self.vIdx = None
		self.numvIdx = 0
		

class SGeomData(Structure):
	_fields_ = [("vVtx",POINTER(MyVec3f)),("vNrm",POINTER(MyVec3f)),("vCol",POINTER(MyVec4f)),("vFace",POINTER(SFace)),("vMap",POINTER(SMappingChannel)),
				("numvVtx",c_int),("numvNrm",c_int),("numvCol",c_int),("numvFace",c_int),("numvMap",c_int)]
	def __init__(self):
		self.vVtx = None
		self.numvVtx = 0
		self.vNrm = None
		self.numvNrm = 0
		self.vCol = None
		self.numvCol = 0
		self.vFace = None
		self.numvFace = 0
		self.vMap = None
		self.numvMap = 0

class MyNode(Structure):
	_fields_ = [("nID",c_int),("nParentID",c_int),("nMaterialIndex",c_int),("sName",c_char_p),("eNodeType",c_int),
				("sNode",SNodeTM),("sGeomData",SGeomData),("vAnim",POINTER(SNodeTM)),("numvAnim",c_int),("sLightData",SLightData),("sCameraData",SCameraData),
				("vModData",POINTER(SModifier)),("numvModData",c_int)]
	def __init__(self):
		self.nID					= 0
		self.nParentID				= -1
		self.nMaterialIndex			= -1
		self.sName 					= ""
		self.eNodeType				= 0
		self.sNode					= SNodeTM()
		self.sGeomData				= SGeomData()
		self.sLightData				= SLightData()
		self.sCameraData			= SCameraData()
				
		self.vModData = None
		self.numvModData = 0
		self.vAnim = None
		self.numvAnim = 0

# TODO: proper texture picking not first enabled

def addSceneMaterials(pvrgp):
	tobeadded = MyMaterial()
	matlist = Material.Get()
	
	for mat in matlist:
		# get the list of txtures for given material
		mattextures = mat.getTextures()
		tobeadded.sName = c_char_p(mat.getName())
		tobeadded.fOpacity = c_float(mat.getAlpha())
		# TODO: check if hardness is ok		
		tobeadded.fGlossiness = c_float((mat.getHardness() -1) /float(510))
		# TODO check if specular level is fine
		tobeadded.fSpecularLevel = c_float(mat.getSpec())
		tobeadded.vDiffuse.x = c_float(mat.getRGBCol()[0])
		tobeadded.vDiffuse.y = c_float(mat.getRGBCol()[1])
		tobeadded.vDiffuse.z = c_float(mat.getRGBCol()[2])
		# World Ambient is multiplied by material ambient factor ,TODO:  is it right?
		tobeadded.vAmbient.x = c_float(World.GetCurrent().getAmb()[0] * mat.getAmb())	 	
		tobeadded.vAmbient.y = c_float(World.GetCurrent().getAmb()[1] * mat.getAmb())	 	
		tobeadded.vAmbient.z = c_float(World.GetCurrent().getAmb()[2] * mat.getAmb())
		tobeadded.vSpecular.x = c_float(mat.getSpecCol()[0])
		tobeadded.vSpecular.y = c_float(mat.getSpecCol()[1])
		tobeadded.vSpecular.z = c_float(mat.getSpecCol()[2])
		
	
		# and so far first active texture is being assumed to be diffuse texture (if any)
		if  len(mat.enabledTextures) > 0 and mat.getTextures()[mat.enabledTextures[0]].tex.getImage() != None:
			tobeadded.sTexDiffuse = c_char_p(mat.getTextures()[mat.enabledTextures[0]].tex.getImage().getName())
			print "Texture attached: ",tobeadded.sTexDiffuse
		else:
			print "No texture exported"
			tobeadded.sTexDiffuse = c_char_p("")
		# TODO: other textures if avilable		
		tobeadded.sTexAmbient = c_char_p("")
		tobeadded.sTexSpecularColour = c_char_p("")
		tobeadded.sTexSpecularLevel = c_char_p("")
		tobeadded.sTexBump = c_char_p("")
		tobeadded.sTexEmissive = c_char_p("")
		tobeadded.sTexGlossiness = c_char_p("")
		tobeadded.sTexOpacity = c_char_p("")
		tobeadded.sTexReflection = c_char_p("")
		tobeadded.sTexRefraction = c_char_p("")
		tobeadded.sPFXFileName = c_char_p("")
		tobeadded.sPFXEffectName = c_char_p("")
		pvrgp.AddMaterial(byref(tobeadded))

def addSceneLamp(pvrgp,obiekt,parentID):
	global uniqueID
	tobefilled = MyNode()
	returnvalue = -1

	if obiekt.getType() == "Lamp":		
		lamp = obiekt.getData()
		tobefilled.nID					= uniqueID	# ujique ID
		returnvalue  					= uniqueID
		uniqueID+=1
		tobefilled.nParentID			= parentID
		tobefilled.nMaterialIndex		= -1
		tobefilled.sName 				= lamp.name
		tobefilled.eNodeType			= eLightCNode
		tobefilled.sNode				= SNodeTM()
		tobefilled.sGeomData			= SGeomData()
		tobefilled.sLightData			= SLightData()
		tobefilled.sCameraData			= SCameraData()
		tobefilled.vModData = None
		tobefilled.numvModData = 0
		tobefilled.vAnim = None
		tobefilled.numvAnim = 0

		if lamp.type == Lamp.Types["Spot"]:
			tobefilled.sLightData.eType = c_int(ePODSpot)
			tobefilled.sLightData.fFalloffAngle		   = c_float(radians(lamp.getSpotSize()))
			tobefilled.sLightData.fFalloffExponent	   = c_float(0.0)			
			tobefilled.sLightData.fConstantAttenuation  = c_float(1.0 - lamp.energy)
			tobefilled.sLightData.fLinearAttenuation    = c_float( (0.5 - (1.0 - lamp.energy))/lamp.dist)
			tobefilled.sLightData.fQuadraticAttenuation = c_float(0.0)
		elif lamp.type == Lamp.Types["Lamp"]:	# Point
			print "ePODPoint: ",ePODPoint
			tobefilled.sLightData.eType = c_int(ePODPoint)
			tobefilled.sLightData.fFalloffAngle		   = c_float(radians(180))
			tobefilled.sLightData.fFalloffExponent	   = c_float(0.0)			
			tobefilled.sLightData.fConstantAttenuation  = c_float(1.0 - lamp.energy)
			tobefilled.sLightData.fLinearAttenuation    = c_float( (0.5 - (1.0 - lamp.energy))/lamp.dist)
			tobefilled.sLightData.fQuadraticAttenuation = c_float(0.0)
			
		elif lamp.type == Lamp.Types["Sun"]:
			tobefilled.sLightData.eType = c_int(ePODDirectional)
			tobefilled.sLightData.fFalloffAngle		   = c_float(radians(180))
			tobefilled.sLightData.fFalloffExponent	   = c_float(0.0)			
			tobefilled.sLightData.fConstantAttenuation  = c_float(1.0)
			tobefilled.sLightData.fLinearAttenuation    = c_float(0.0)
			tobefilled.sLightData.fQuadraticAttenuation = c_float(0.0)
		else:
			return 
	
		tobefilled.vColour = MyVec3f(lamp.R,lamp.G,lamp.B)
		tobefilled.nTargetID = -1 # no target light source

		mv = obiekt.getMatrix("localspace")
		# do the conversion if only there is no parent (final transformation is multiplied by one)
		if parentID == -1:
			nmv = mat4x4DirectX2OpenGLCoords(mv.copy())
		else:
			nmv = mv.copy()
		matrixToSNodeTM(tobefilled.sNode, nmv)		
		# copy transformation matrix of the current frame which should be frame 1
		Set('curframe',1)
		for i in range(0,4):
			for j in range(0,4):
				tobefilled.sNode.fMatrix.f[i*4+j] = nmv[i][j]			
				
		# IPO (animtaion) if any
		if export_options.exportControllers == 1 and obiekt.getIpo() != None:
			# Go through all frames and get modelview matrices for this object
			startframe = Get('staframe')
			endframe = Get('endframe')
			# allocate space for animation data	(number)
			tobefilled.numvAnim = c_int(endframe)
			tobefilled.vAnim = (endframe*SNodeTM)()
			for fridx in range(0,endframe):
				tobefilled.vAnim[fridx] = SNodeTM()
			
			for framenum in range(startframe,endframe+1):
				Set('curframe',framenum)
				mv = obiekt.getMatrix("localspace")
				if parentID == -1:
					nmv = mat4x4DirectX2OpenGLCoords(mv.copy())
				else:
					nmv = mv.copy()
				for i in range(0,4):
					for j in range(0,4):
							tobefilled.vAnim[framenum-1].fMatrix.f[i*4+j] = nmv[i][j]
				matrixToSNodeTM(tobefilled.vAnim[framenum-1], nmv)						
		# pass it to POD 
		pvrgp.AddNode(byref(tobefilled))				
	return returnvalue
		
def addSceneCamera(pvrgp,obiekt,parentID):
	global uniqueID
	global export_options
	
	tobefilled = MyNode()

	returnvalue	= -1
	if obiekt.getType() == "Camera":
		camera = obiekt.getData()
		if camera.type == "persp":

			tobefilled.nID					= 	uniqueID
			returnvalue  					= uniqueID
			uniqueID+=1
			
			tobefilled.nParentID				= parentID
			tobefilled.nMaterialIndex			= -1
			tobefilled.sName 					= camera.name
			tobefilled.eNodeType				= eCameraCNode
			tobefilled.sNode					= SNodeTM()
			tobefilled.sGeomData				= SGeomData()
			tobefilled.sLightData				= SLightData()
			tobefilled.sCameraData				= SCameraData()
			tobefilled.vModData = None
			tobefilled.numvModData = 0
			tobefilled.vAnim = None
			tobefilled.numvAnim = 0
			
			# CamerData should be filled in
			tobefilled.sCameraData.fFOV = c_float(radians(camera.angle))
			tobefilled.sCameraData.fFarClip = c_float(camera.clipEnd)
			tobefilled.sCameraData.fNearClip = c_float(camera.clipStart)
			# TODO: target camera
			tobefilled.sCameraData.nTargetID = -1 # free camera 

			# OpenGL coords require to be rotated to be exported to POD			
			# For POD we need to change coordinates system.
			# So we rotate by 90,180,180 and then camera rotation angles to be applied
			# also are tranformed x => -x (why minus?), y => z , z => -y			
			# NOTE: do not used setMatrix method becouse it does not update IPO.. however getMatrix return final matrix
			# so executing : mat = obiekt.getMatrix(`localspace`) obiekt.setMatrix(`localspace`)

#			print "camera angles: x=",obiekt.getEuler().x*180.0/3.141592653," y=",obiekt.getEuler().y*180.0/3.141592653," z=",obiekt.getEuler().z*180.0/3.141592653
			mv = obiekt.getMatrix("localspace")
			if parentID == -1:
				nmv = mat4x4DirectX2OpenGLCoords(mv.copy())
			else:
				nmv = mv.copy()			
			matrixToSNodeTM(tobefilled.sNode, nmv)		
			# copy transformation matrix of the current frame which should be frame 1
			Set('curframe',1)
			for i in range(0,4):
				for j in range(0,4):
					tobefilled.sNode.fMatrix.f[i*4+j] = nmv[i][j]			
				
			# IPO (animtaion) if any
			if export_options.exportControllers == 1 and obiekt.getIpo() != None:
				# Go through all frames and get modelview matrices for this object
				startframe = Get('staframe')
				endframe = Get('endframe')
				# allocate space for animation data	(number)
				tobefilled.numvAnim = c_int(endframe)
				tobefilled.vAnim = (endframe*SNodeTM)()
				for fridx in range(0,endframe):
					tobefilled.vAnim[fridx] = SNodeTM()
				
				for framenum in range(startframe,endframe+1):
					Set('curframe',framenum)
					# again. get original angles, clear coords (return camera to parentspace), roate coords, 
					#apply original transformation in another coords system
					# values to be restores are returned as well as new modelview
					mv = obiekt.getMatrix("localspace")
					if parentID == -1:
						nmv = mat4x4DirectX2OpenGLCoords(mv.copy())
					else:
						nmv = mv.copy()
					for i in range(0,4):
						for j in range(0,4):
								tobefilled.vAnim[framenum-1].fMatrix.f[i*4+j] = nmv[i][j]
					matrixToSNodeTM(tobefilled.vAnim[framenum-1], nmv)						
			else:
				print "NO CAMERA IPO"								
			# pass it to POD
			pvrgp.AddNode(byref(tobefilled))		
		else:
			print "Warning: Orthogonal Camera will not be exported!"
	return returnvalue
def showListOfAddedObjects():
	for i in listOfAddedObjects:
		print "obj: ",i," id=",listOfAddedObjects[i]
	return

# This function adds dummy based on given object whatever type
# basically we take parenting info and modelview transformation and we create
# dummy object holding that info. Animation included as well
# ID is assigned based on the value that was passed to that function
def addSceneDummyOrBone(pvrgp,obiekt,typobiektu,tobeassignedID,parentID):
	global export_options
	
	tobefilled = MyNode()
	returnvalue	= -1

	obiektname = obiekt.getName()
	tobefilled.nID					= tobeassignedID
	returnvalue  					= tobeassignedID
	
	tobefilled.nParentID				= parentID
	tobefilled.nMaterialIndex			= -1
	tobefilled.sName 					= obiektname
	tobefilled.eNodeType				= typobiektu #eDummyCNode,eBoneCNode
	tobefilled.sNode					= SNodeTM()
	tobefilled.sGeomData				= SGeomData()
	tobefilled.sLightData				= SLightData()
	tobefilled.sCameraData				= SCameraData()
	tobefilled.vModData = None
	tobefilled.numvModData = 0
	tobefilled.vAnim = None
	tobefilled.numvAnim = 0


	mv = obiekt.getMatrix("localspace")
	
	if parentID == -1:
		nmv = mat4x4DirectX2OpenGLCoords(mv.copy())
	else:
		nmv = mv.copy()	
	matrixToSNodeTM(tobefilled.sNode, nmv)		
	# copy transformation matrix of the current frame which should be frame 1
	Set('curframe',1)
	for i in range(0,4):
		for j in range(0,4):
			tobefilled.sNode.fMatrix.f[i*4+j] = nmv[i][j]			

	# IPO (animtaion) if any
	if export_options.exportControllers == 1 and obiekt.getIpo() != None:
		# Go through all frames and get modelview matrices for this object
		startframe = Get('staframe')
		endframe = Get('endframe')
		# allocate space for animation data	(number)
		tobefilled.numvAnim = c_int(endframe)
		tobefilled.vAnim = (endframe*SNodeTM)()
		for fridx in range(0,endframe):
			tobefilled.vAnim[fridx] = SNodeTM()
		
		for framenum in range(startframe,endframe+1):
			Set('curframe',framenum)
			# again. get original angles, clear coords (return camera to parentspace), roate coords, 
			#apply original transformation in another coords system
			# values to be restores are returned as well as new modelview
			mv = obiekt.getMatrix("localspace")
			if parentID == -1:
				nmv = mat4x4DirectX2OpenGLCoords(mv.copy())
			else:
				nmv = mv.copy()			
			for i in range(0,4):
				for j in range(0,4):
						tobefilled.vAnim[framenum-1].fMatrix.f[i*4+j] = nmv[i][j]
			matrixToSNodeTM(tobefilled.vAnim[framenum-1], nmv)						
			
	else:
		print "no IPO for DUMMY"								
	# pass it to POD
	pvrgp.AddNode(byref(tobefilled))		
	return returnvalue

def addSceneRelativesBones(pvrgp,chainObjects,arm_obiekt,parentid):
	global uniqueID
	global listOfAddedObjects
	parID = parentid
	tobefilled = MyNode()
	returnvalue = -1
	# the eldest come first
	
	for bone in chainObjects:
		tobefilled.nID					= uniqueID
		tobefilled.nParentID			= parID
		listOfAddedObjects[bone.name]	= uniqueID
		parID 							= uniqueID
		uniqueID+=1
		tobefilled.nMaterialIndex			= -1
		tobefilled.sName 					= bone.name
		tobefilled.eNodeType				= eBoneCNode
		tobefilled.sNode					= SNodeTM()
		tobefilled.sGeomData				= SGeomData()
		tobefilled.sLightData				= SLightData()
		tobefilled.sCameraData				= SCameraData()
		tobefilled.vModData = None
		tobefilled.numvModData = 0
		tobefilled.vAnim = None
		tobefilled.numvAnim = 0

		print "Adding Bone: ",bone.name," ID=",tobefilled.nID," parentID=",tobefilled.nParentID

		# copy transformation matrix of the current frame which should be frame 1
		Set('curframe',1)
		arm_obiekt.evaluatePose(1)
		# Transformations data is filled below
		# We need an transformation according the parent coords (bone space?) coords
		mv = bone.poseMatrix #* arm_obiekt.getMatrix()
		# since we are given ulitmate pose matrix (once includeing parents transofmraion )
		# then we need to transform it back to have a transformation in parent coords not in armature coords (if only there is a parent)
		# The matrix is transposed comparing to normal transformation matrix so multiplication 
		# takes place in reverse order		
		mvc = mv.copy()
		if bone.parent != None:
			pmv = bone.parent.poseMatrix
			#mvc = pmv.invert()*mvc
			mvc = mvc*pmv.invert()
		nmv = mvc
		
		# revese z value
		matrixToSNodeTM(tobefilled.sNode, nmv)		
		for i in range(0,4):
			for j in range(0,4):
				tobefilled.sNode.fMatrix.f[i*4+j] = nmv[i][j]
		# Go through all frames and get modelview matrices for this object
		startframe = Get('staframe')
		endframe = Get('endframe')
		# allocate space for animation data	(number)
		tobefilled.numvAnim = c_int(endframe)
		tobefilled.vAnim = (endframe*SNodeTM)()
		for fridx in range(0,endframe):
			tobefilled.vAnim[fridx] = SNodeTM()
		
#		print "Armature MV Worldmat: ",arm_obiekt.getMatrix()
		
		for framenum in range(startframe,endframe+1):
			Set('curframe',framenum)
			arm_obiekt.evaluatePose(framenum)
	#		print "=============================================="
	#		print "frame: ",framenum," globposemat: ",bone.poseMatrix
				
			mv = bone.poseMatrix  #* arm_obiekt.getMatrix()
			mvc = mv.copy()
			if bone.parent != None:
				pmv = bone.parent.poseMatrix
				#print "frame: ",framenum," pose: ",bone.poseMatrix*pmv.invert()
				#mvc = pmv.invert()*mvc
				mvc = mvc*pmv.invert()
			nmv = mvc
#			print "frame: ",framenum," local bone matrix: ",mvc
			
			for i in range(0,4):
				for j in range(0,4):
					tobefilled.vAnim[framenum-1].fMatrix.f[i*4+j] = nmv[i][j]
			matrixToSNodeTM(tobefilled.vAnim[framenum-1], nmv)			
		Set('curframe',1)		
		arm_obiekt.evaluatePose(1)
		# Get material index and store it this node	
		pvrgp.AddNode(byref(tobefilled))	
	# last one so, the requested bone to be added (rest bones in chain are elder that that)
	return parID	

def addSceneBones(pvrgp,obiekt,parentID):
	global uniqueID
	global listOfAddedObjects
	chainOfBones = []
	returnvalue	= -1
	armatureID = parentID
	if obiekt.getType() == "Armature":

		print "Adding Armature Bones , armatureID=",armatureID
		
		pbones = obiekt.getPose().bones	
		for bone in pbones.keys(): #armature.bones.keys():
			tobefilled = MyNode()
			# check if bone was already added
			if not listOfAddedObjects.has_key(bone):
				# now check if given object got a parent, parent has to be added first
				# so find first ancastor not added that does not have any objects
				# create a list of ancastors to be added in the order starting from the eldest
				tmpbone = bone
				while pbones[tmpbone].parent != None  and (not listOfAddedObjects.has_key(pbones[tmpbone].parent.name)):
					chainOfBones.insert(0,pbones[tmpbone])	
					tmpbone = pbones[tmpbone].parent.name
				chainOfBones.insert(0,pbones[tmpbone])
				if pbones[tmpbone].parent == None:
					parentID = armatureID
				else:
					parentID = listOfAddedObjects[pbones[tmpbone].parent.name]
				print "chainsOFBones to be added: ",chainOfBones	
				addSceneRelativesBones(pvrgp,chainOfBones,obiekt,parentID)				
			chainOfBones = []

		# tutaj sobie wypisze info jakies

	# Dodawanie kosteczki
	return returnvalue

def fillInSkinningData(name_of_object,tobefilled,moddata_index,vert_idx,influences,numInfluences):
	dictOfInfluences = {}
	
	if numInfluences == 0:
		return
	for bone, weight in influences:
#		print "Influence: id: ",bone," weight: ",weight
		dictOfInfluences[bone] = weight
		
	# for more that four bones, sort all in order of decreasing values and get rid of any but for entries with biggest values		
	if numInfluences > 4:
		print "Warning: More than for bones influencing the vertex! Performing striping less inflencial influences"
		listOfInfluences = sorted(dictOfInfluences.items(),key=itemgetter(1),reverse=True)  
		del listOfInfluences[4:]
		for bone,weight in listOfInfluences:
			dictOfInfluences[bone]=weight
	global listOfAddedObjects
	# calculate the sum of weights
	sumOfWeights = 0
	for bone in dictOfInfluences:
		sumOfWeights+=dictOfInfluences[bone]
	# perform normalization
	# For sum of weights close to zero then no normalization will take place
#	print "Skiining vert_indx=",vert_idx," lenOfInfluences=",len(dictOfInfluences)
#	print "Skiining vert[",vert_idx,"]=",tobefilled.vModData[moddata_index].vVtx[vert_idx]
	tobefilled.vModData[moddata_index].vVtx[vert_idx].numvBone = len(dictOfInfluences)
	tobefilled.vModData[moddata_index].vVtx[vert_idx].vBone = (len(dictOfInfluences)*CBoneWeight)()
	for boneweightidx in range(0,len(dictOfInfluences)):
		tobefilled.vModData[moddata_index].vVtx[vert_idx].vBone[boneweightidx] = CBoneWeight()
	
	# For sum of weights close to zero then no normalization will take place 
	if sumOfWeights < MAXIMAL_VARIATION:
		sumOfWeights = 1.0	
	lidx = 0
	for bone in dictOfInfluences:
		dictOfInfluences[bone]/=sumOfWeights
		if not listOfAddedObjects.has_key(bone):
			print "Warning: Mesh (\"",name_of_object,"\") is influenced by vertex group referring to bones that does not exist. Skinning is NOT exported"
			return			
		tobefilled.vModData[moddata_index].vVtx[vert_idx].vBone[lidx].nBone = listOfAddedObjects[bone]
		tobefilled.vModData[moddata_index].vVtx[vert_idx].vBone[lidx].fWeight = dictOfInfluences[bone]
#		print "filled entry: ",tobefilled.vModData[moddata_index].vVtx[vert_idx].vBone[lidx].nBone," weight=",tobefilled.vModData[moddata_index].vVtx[vert_idx].vBone[lidx].fWeight
		lidx+=1
##	print "local influences: ",dictOfInfluences," len of dict: ",len(dictOfInfluences)
	return	
		
def addSceneMesh(pvrgp,obiekt,parentID):
	global uniqueID
	global export_options

	returnvalue	= -1
	tobefilled = MyNode()

	editmode = Window.EditMode()
	if editmode == 1:
		Window.EditMode(0)


	if obiekt.getType() == "Mesh":
		print "Obiekt:  "+obiekt.getName()
		mesik_orig = obiekt.getData(False,True)
		mesik = mesik_orig.__copy__()
		# POD support only faces that are triangles
		# converting quads into triangles works only in "face mode"
		# which we need to set for that moment
		
		# Get copy of mesh, create new object base on copy of mesh
		# so we can freely modify this copy , preserving original scene
		orig_scene = Scene.GetCurrent()
		tmpobiekt = orig_scene.objects.new(mesik)

		mesik.sel = True

	
		oldmode = Mesh.Mode()
		Mesh.Mode(Mesh.SelectModes['FACE'])
		mesik.quadToTriangle(0)
		Mesh.Mode(oldmode)
		tobefilled.nID					= uniqueID
		returnvalue  					= uniqueID
		uniqueID+=1

	
		tobefilled.nParentID				= parentID
		tobefilled.nMaterialIndex			= -1
		tobefilled.sName 					= obiekt.getName()
		tobefilled.eNodeType				= eMeshCNode
		tobefilled.sNode					= SNodeTM()
		tobefilled.sGeomData				= SGeomData()
		tobefilled.sLightData				= SLightData()
		tobefilled.sCameraData				= SCameraData()
		tobefilled.vModData = None
		tobefilled.numvModData = 0
		tobefilled.vAnim = None
		tobefilled.numvAnim = 0

		# Number of stored vertices will be 3 times number of faces
		# it is huge redundancy that will be later reduced during POD creation process
		tobefilled.sGeomData.numvVtx = c_int(len(mesik.faces)*3)
		tobefilled.sGeomData.vVtx = (len(mesik.faces)*3* MyVec3f)()
		for vidx in range(0,len(mesik.faces)*3):
			tobefilled.sGeomData.vVtx[vidx] = MyVec3f(0,0,0)
		
		if export_options.exportNormals:
			tobefilled.sGeomData.numvNrm = c_int(len(mesik.faces)*3)
			tobefilled.sGeomData.vNrm = (len(mesik.faces)*3* MyVec3f)()
			for vidx in range(0,len(mesik.faces)*3):
				tobefilled.sGeomData.vNrm[vidx] = MyVec3f(0,0,0)


		if mesik.faceUV == True and export_options.exportMappingChannel: 
			# amount of entries is 3 per every face (triangle) , becaouse we
			# grab texcords per face
			print "Exporting mapping channels"			
			tobefilled.sGeomData.numvMap = 1	# To be changed when supporting many channels
			tobefilled.sGeomData.vMap = (tobefilled.sGeomData.numvMap*SMappingChannel)()
			for i in range(0,tobefilled.sGeomData.numvMap):
				tobefilled.sGeomData.vMap[i].numvVtx = c_int(len(mesik.faces)*3)
				tobefilled.sGeomData.vMap[i].vVtx = (len(mesik.faces)*3*MyVec3f)()
				for vidx in range(0,len(mesik.faces)*3):
					tobefilled.sGeomData.vMap[i].vVtx[vidx] = MyVec3f(0,0,0)
				tobefilled.sGeomData.vMap[i].numvIdx = c_int(len(mesik.faces))
				tobefilled.sGeomData.vMap[i].vIdx = (len(mesik.faces)*CTriIdx)()					
				for idx in range(0,len(mesik.faces)):
					tobefilled.sGeomData.vMap[i].vIdx[idx] = CTriIdx()					
				

		tobefilled.sGeomData.numvFace = c_int(len(mesik.faces))
		tobefilled.sGeomData.vFace = (len(mesik.faces) * SFace)()
		for idx in range(0,len(mesik.faces)):
			tobefilled.sGeomData.vFace[idx] = SFace()

		# three colors per face so total number of colours is numFaces*3 (I knowthat it is  redundant)
		if mesik.vertexColors == True and  export_options.exportVertexColor :
			tobefilled.sGeomData.numvCol = c_int(len(mesik.faces) * 3)  
			tobefilled.sGeomData.vCol = (len(mesik.faces) * 3 *MyVec4f)()
			for idx in range(0,len(mesik.faces)):
				tobefilled.sGeomData.vCol[idx] = MyVec4f()				

		if export_options.exportSkin :
			# TODO: allow more modifiers
			# Copy of mesh does not copy 
#			print "Orig Bone influences: ",mesik_orig.getVertexInfluences(0)
			if len(mesik_orig.getVertexInfluences(0)):
				print "Object(Mesh): ",obiekt.getName()," Skinning exported"
				tobefilled.numvModData = 1
				tobefilled.vModData	= (tobefilled.numvModData*SModifier)()
				for i in range(0,tobefilled.numvModData):
					tobefilled.vModData[i].sName = c_char_p("Skin")
					tobefilled.vModData[i].eSkinType = c_int(ePODSkin)
					tobefilled.vModData[i].bIsSkin = c_int(1)
					tobefilled.vModData[i].numvVtx = tobefilled.sGeomData.numvVtx
					tobefilled.vModData[i].vVtx = (tobefilled.vModData[i].numvVtx*SBonedVtx)()
					for vidx in range(0,tobefilled.sGeomData.numvVtx):
						tobefilled.sGeomData.vVtx[vidx] = MyVec3f(0,0,0)

					
					# space for bones and weights will be assigned later , according to needs
			else:
				print "Object(Mesh): ",obiekt.getName()," No Skinning exported"				


		# Very memory consuming implementation
		# Each face got 3 vertices because mesh was converted to be like that
		fidx = 0
		vidx = 0
		colIdx = 0

		if export_options.exportNormals:
			mesik.calcNormals()

		for face in mesik.faces:
			# grab the blender vertex data 
			v0 = mesik.verts[face.v[0].index]
			v1 = mesik.verts[face.v[1].index]
			v2 = mesik.verts[face.v[2].index]

			# adding new vertices
			tobefilled.sGeomData.vVtx[vidx].x = v0.co.x
			tobefilled.sGeomData.vVtx[vidx].y = v0.co.y
			tobefilled.sGeomData.vVtx[vidx].z = v0.co.z
			tobefilled.sGeomData.vVtx[vidx+1].x = v1.co.x
			tobefilled.sGeomData.vVtx[vidx+1].y = v1.co.y
			tobefilled.sGeomData.vVtx[vidx+1].z = v1.co.z
			tobefilled.sGeomData.vVtx[vidx+2].x = v2.co.x
			tobefilled.sGeomData.vVtx[vidx+2].y = v2.co.y
			tobefilled.sGeomData.vVtx[vidx+2].z = v2.co.z
			# normals 
			if export_options.exportNormals:
				if face.smooth:
					tobefilled.sGeomData.vNrm[vidx].x = v0.no.x
					tobefilled.sGeomData.vNrm[vidx].y = v0.no.y
					tobefilled.sGeomData.vNrm[vidx].z = v0.no.z
					#print "Face smooth normal 0 x=",v0.no.x," y=",v0.no.y," z=",v0.no.z
					#vec = Mathutils.Vector(v0.no.x,v0.no.y,v0.no.z)
					#vec.normalize()
					#print "Face smooth normalized normal 0 x=",vec.x," y=",vec.y," z=",vec.z
					#tobefilled.sGeomData.vNrm[vidx].x = vec.x
					#tobefilled.sGeomData.vNrm[vidx].y = vec.y
					#tobefilled.sGeomData.vNrm[vidx].z = vec.z
					tobefilled.sGeomData.vNrm[vidx+1].x = v1.no.x
					tobefilled.sGeomData.vNrm[vidx+1].y = v1.no.y
					tobefilled.sGeomData.vNrm[vidx+1].z = v1.no.z
					#print "Face smooth normal 1 x=",v1.no.x," y=",v1.no.y," z=",v1.no.z
					#vec = Mathutils.Vector(v1.no.x,v1.no.y,v1.no.z)
					#vec.normalize()
					#print "Face smooth normalized normal 1 x=",vec.x," y=",vec.y," z=",vec.z
					#tobefilled.sGeomData.vNrm[vidx].x = vec.x
					#tobefilled.sGeomData.vNrm[vidx].y = vec.y
					#tobefilled.sGeomData.vNrm[vidx].z = vec.z
					tobefilled.sGeomData.vNrm[vidx+2].x = v2.no.x
					tobefilled.sGeomData.vNrm[vidx+2].y = v2.no.y
					tobefilled.sGeomData.vNrm[vidx+2].z = v2.no.z
					#print "Face smooth normal 2 x=",v1.no.x," y=",v1.no.y," z=",v1.no.z
					#vec = Mathutils.Vector(v2.no.x,v2.no.y,v2.no.z)
					#vec.normalize()
					#print "Face smooth normalized normal 2 x=",vec.x," y=",vec.y," z=",vec.z
					#tobefilled.sGeomData.vNrm[vidx].x = vec.x
					#tobefilled.sGeomData.vNrm[vidx].y = vec.y
					#tobefilled.sGeomData.vNrm[vidx].z = vec.z
					
				else:
					#vec = Mathutils.Vector(face.no.x,face.no.y,face.no.z)
					#vec.normalize()
					#tobefilled.sGeomData.vNrm[vidx].x = vec.x
					#tobefilled.sGeomData.vNrm[vidx].y = vec.y
					#tobefilled.sGeomData.vNrm[vidx].z = vec.z
					#tobefilled.sGeomData.vNrm[vidx+1].x = vec.x
					#tobefilled.sGeomData.vNrm[vidx+1].y = vec.y
					#tobefilled.sGeomData.vNrm[vidx+1].z = vec.z
					#tobefilled.sGeomData.vNrm[vidx+2].x = vec.x
					#tobefilled.sGeomData.vNrm[vidx+2].y = vec.y
					#tobefilled.sGeomData.vNrm[vidx+2].z = vec.z
					tobefilled.sGeomData.vNrm[vidx].x = face.no.x
					tobefilled.sGeomData.vNrm[vidx].y = face.no.y
					tobefilled.sGeomData.vNrm[vidx].z = face.no.z
					tobefilled.sGeomData.vNrm[vidx+1].x = face.no.x
					tobefilled.sGeomData.vNrm[vidx+1].y = face.no.y
					tobefilled.sGeomData.vNrm[vidx+1].z = face.no.z
					tobefilled.sGeomData.vNrm[vidx+2].x = face.no.x
					tobefilled.sGeomData.vNrm[vidx+2].y = face.no.y
					tobefilled.sGeomData.vNrm[vidx+2].z = face.no.z
					
			# Only up to 4 bones per vertex are supported
			if export_options.exportSkin and len(mesik_orig.getVertexInfluences(0)):
				influences = mesik_orig.getVertexInfluences(face.v[0].index)
				fillInSkinningData(obiekt.getName(),tobefilled,0,vidx+0,influences,len(influences),)
				influences = mesik_orig.getVertexInfluences(face.v[1].index)
				fillInSkinningData(obiekt.getName(),tobefilled,0,vidx+1,influences,len(influences),)
				influences = mesik_orig.getVertexInfluences(face.v[2].index)
				fillInSkinningData(obiekt.getName(),tobefilled,0,vidx+2,influences,len(influences),)
				
			# fill the faces data with our newly created vertex indices						
			tobefilled.sGeomData.vFace[fidx].sVtx.i[0] = vidx
			tobefilled.sGeomData.vFace[fidx].sVtx.i[1] = vidx + 1
			tobefilled.sGeomData.vFace[fidx].sVtx.i[2] = vidx + 2
			# Normals to vertices correspond to the indices 
			if export_options.exportNormals:
				tobefilled.sGeomData.vFace[fidx].sNrm.i[0] = vidx
				tobefilled.sGeomData.vFace[fidx].sNrm.i[1] = vidx + 1
				tobefilled.sGeomData.vFace[fidx].sNrm.i[2] = vidx + 2
			# TODO: colours to be tested
			if mesik.vertexColors == True and export_options.exportVertexColor:
				# add a color to the list of colors
				tobefilled.sGeomData.vCol[colIdx].x = face.col[0].r
				tobefilled.sGeomData.vCol[colIdx].y = face.col[0].g
				tobefilled.sGeomData.vCol[colIdx].z = face.col[0].b
				tobefilled.sGeomData.vCol[colIdx].w = face.col[0].a

				tobefilled.sGeomData.vCol[colIdx+1].x = face.col[1].r
				tobefilled.sGeomData.vCol[colIdx+1].y = face.col[1].g
				tobefilled.sGeomData.vCol[colIdx+1].z = face.col[1].b
				tobefilled.sGeomData.vCol[colIdx+1].w = face.col[1].a

				tobefilled.sGeomData.vCol[colIdx+2].x = face.col[2].r
				tobefilled.sGeomData.vCol[colIdx+2].y = face.col[2].g
				tobefilled.sGeomData.vCol[colIdx+2].z = face.col[2].b
				tobefilled.sGeomData.vCol[colIdx+2].w = face.col[2].a
				
				# assign color index into POD face data
				tobefilled.sGeomData.vFace[fidx].sVtxCol.i[0] = colIdx
				tobefilled.sGeomData.vFace[fidx].sVtxCol.i[1] = colIdx+1
				tobefilled.sGeomData.vFace[fidx].sVtxCol.i[2] = colIdx+2
				colIdx+=3
			# UV coords 
			if  mesik.faceUV == True and export_options.exportMappingChannel == 1:
				#print "exportuje UVcoords"
				fillMappingChannelEntry(tobefilled.sGeomData.vMap[0],face,fidx,vidx)
				#print "exported uvs x= ",tobefilled.sGeomData.vMap[0].vVtx[vidx].x," y= ",tobefilled.sGeomData.vMap[0].vVtx[vidx].y," idx=",fidx
			fidx+=1
			vidx+=3
		
		# Transofmations data is filled below
		# We need an transformation according the local coords
		mv = obiekt.getMatrix("localspace")
		# revese z value
		if parentID == -1:
			nmv = mat4x4DirectX2OpenGLCoords(mv.copy())
		else:
			nmv = mv.copy()		
		#print "object being stored: ",tobefilled.sName
		#print "object transformation matrix: ",nmv
		#print "object rotation part of matrix: ",nmv.rotationPart()
		#print "object Euler of matrix: ",nmv.toEuler()
		#newmatrix = nmv.copy()
		#newmatrix.transpose()
		#print "object Euler of transposed matrix: ",newmatrix.toEuler()
		#print "object quaternion:",nmv.toQuat()
		#q = nmv.toEuler().toQuat()
		#print "object euler->quaternion:",q, "angle=",q.angle," axis[0]=",q.axis[0]," axis[1]=",q.axis[1]," axis[2]=",q.axis[2]
		#print "object quat->matrix",q.toMatrix()
	
		matrixToSNodeTM(tobefilled.sNode, nmv)		

		# copy transformation matrix of the current frame which should be frame 1
		Set('curframe',1)
		for i in range(0,4):
			for j in range(0,4):
				tobefilled.sNode.fMatrix.f[i*4+j] = nmv[i][j]

		# IPO (animtaion)
		objIpo = obiekt.getIpo()
		if objIpo != None:
			# Go through all frames and get modelview matrices for this object
			startframe = Get('staframe')
			endframe = Get('endframe')
			
			# allocate space for animation data	(number)
			tobefilled.numvAnim = c_int(endframe)
			tobefilled.vAnim = (endframe*SNodeTM)()
			for fridx in range(0,endframe):
				tobefilled.vAnim[fridx] = SNodeTM()
			for framenum in range(startframe,endframe+1):
				Set('curframe',framenum)
				mv = obiekt.getMatrix("localspace")
				# revese z value
				if parentID == -1:
					nmv = mat4x4DirectX2OpenGLCoords(mv.copy())
				else:
					nmv = mv.copy()				
				for i in range(0,4):
					for j in range(0,4):
						tobefilled.vAnim[framenum-1].fMatrix.f[i*4+j] = nmv[i][j]
				matrixToSNodeTM(tobefilled.vAnim[framenum-1], nmv)			
				
		# Get material index and store it this node	
		tobefilled.nMaterialIndex = c_int(getMaterialIndex(obiekt.activeMaterial))
		# revert triangulation		

		orig_scene.objects.unlink(tmpobiekt)		

		pvrgp.AddNode(byref(tobefilled))		
	if editmode == 1:
		Window.EditMode(1)
		
	return returnvalue

def addSceneRelativesObjects(pvrgp,chainObjects,listOfAddedArmatures,parentid):
	# go throuh all of given object , that should be in 
	# the order of obiekt, child, child of child etc...
	global listOfAddedObjects
	global uniqueID
	
	parID = parentid
	for obiekt in chainObjects:
		# add object
		if obiekt.getType() == "Mesh":
			#print "Mesh: ",obiekt.getName(),", parent: ",obiekt.getParent()," parID=",parID
			listOfAddedObjects[obiekt.getName()] = addSceneMesh(pvrgp,obiekt,parID)
			parID = listOfAddedObjects[obiekt.getName()]
			#print "list: ",listOfAddedObjects
		elif obiekt.getType() == "Camera":
			#print "Camera: ",obiekt.getName(),", parent: ",obiekt.getParent()
			listOfAddedObjects[obiekt.getName()] = addSceneCamera(pvrgp,obiekt,parID)
			parID = listOfAddedObjects[obiekt.getName()]
			#print "list: ",listOfAddedObjects
		elif obiekt.getType() == "Lamp":
			#print "Lamp: ",obiekt.getName(),", parent: ",obiekt.getParent()
			listOfAddedObjects[obiekt.getName()] = addSceneLamp(pvrgp,obiekt,parID)
			parID = listOfAddedObjects[obiekt.getName()]
			#print "list: ",listOfAddedObjects
		elif  obiekt.getType() == "Armature":
			# Here we add dummy object holding only transformation and relationships info
			#print "Armature: ",obiekt.getName(),", parent: ",obiekt.getParent()
			# here find the given armature on the list of IDs of armatures and
			# pass ID to function creating dummy
			tobeassignedID = listOfAddedArmatures[obiekt.getName()]  
			listOfAddedObjects[obiekt.getName()] = addSceneDummyOrBone(pvrgp,obiekt,eBoneCNode,tobeassignedID,parID)
			parID = listOfAddedObjects[obiekt.getName()]
			#print "list: ",listOfAddedObjects
		elif obiekt.getType() == "Empty":
			tobeassignedID = uniqueID			
			uniqueID += 1
			listOfAddedObjects[obiekt.getName()] = addSceneDummyOrBone(pvrgp,obiekt,eDummyCNode,tobeassignedID,parID)
		else:
			print "Another Kind of object: ",obiekt.getType()," name: ",obiekt.getName(),", parent: ",obiekt.getParent()

def addSceneObjects(pvrgp):

	global listOfAddedObjects
	global uniqueID
	
	# export z aktywnej sceny
	orig_scene = Scene.GetCurrent()

	listOfAddedObjects = {}
	chainOfObjects = []

	listOfAddedArmatures = {}

	# first passing to get armatures nd add all bones
	# TODO: implement consideration of parents of armatures. Only armature can parent other armature 
	for obiekt in orig_scene.objects:
		if obiekt.getType() == "Armature":
			if obiekt.getData().vertexGroups == True:
				listOfAddedArmatures[obiekt.getName()] = uniqueID
				uniqueID += 1
				addSceneBones(pvrgp,obiekt,listOfAddedArmatures[obiekt.getName()])				
				# here we record ID passed as a parent of root bone and later the same one will be used
				# for ID of dummy reperesenting Armature object (its World View matrix)
			if obiekt.getData().envelopes == True:
				print "Warning: Export of Envelopes based skinning not implemented!"

	# second passing so that bones are already passed and its IDs are known
	for obiekt in orig_scene.objects:
		# check if object was already added
		#print "OBIEKT to be analyzed: ",obiekt.getName()
		if not listOfAddedObjects.has_key(obiekt.getName()):
			#print "obiekt: ",obiekt.getName()," parent: ",obiekt.getParent()," not added before. Adding!"
			# now check if given object got a parent (Armature objects do not count), parent has to be added first
			# so find first ancastor not added that does not have any objects
			# create a list of ancastors to be added in the order starting from the eldest
			chainOfObjects.append(obiekt)
			tmpobiekt = obiekt
			while tmpobiekt.getParent() != None and (not listOfAddedObjects.has_key(tmpobiekt.getParent().getName())):
				chainOfObjects.insert(0,tmpobiekt)	
				tmpobiekt = tmpobiekt.getParent()
			if tmpobiekt.getParent() == None:
				parentID = -1
			else:
				parentID = listOfAddedObjects[tmpobiekt.getParent().getName()]
			addSceneRelativesObjects(pvrgp,chainOfObjects,listOfAddedArmatures,parentID)				
			chainOfObjects = []
			
	# Now some of the bones are parented by Armatures , which is my dummy objects. So we need to 
	# update its parents with IDs of dummies
	#for obiekt in listOfAddedObjects:
		
	return 

# get ID of proper Material for given object
def getMaterialIndex(activematindex):
	global export_options

	
	# if activematindex is zero then material not assigned
	# other wise value should decrease by 1
	# if no export of materials was chosen then also put -1
	if export_options.exportMaterials == 1:
		return activematindex - 1
	else:
		return -1


			
# Change blender coords (DirectX like coordinates system) to POD coords( OpenGL likecoordinates system)
# Reverting Z axe is needed
def mat4x4DirectX2OpenGLCoords(mat):
#	mat_flip = Mathutils.Matrix([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1])
#	mat*=mat_flip
#	mat[2][0]*= (-1.0)
#	mat[2][1]*= (-1.0)
#	mat[2][2]*= (-1.0)
#	mat[2][3]*= (-1.0)
	return mat


def matrixToSNodeTM(snode, matrix):
	trans = matrix.translationPart()
	snode.vPos.x = trans[0]
	snode.vPos.y = trans[1]
	snode.vPos.z = trans[2]
	newmatrix = matrix.copy()
	# TODO: find out why transpose is needed here
	newmatrix.transpose()	
	qrotate = newmatrix.toEuler().toQuat()
	snode.qRot.x = qrotate.x
	snode.qRot.y = qrotate.y
	snode.qRot.z = qrotate.z
	snode.qRot.w = qrotate.w
	scale = newmatrix.scalePart()
	snode.sScale.f0 = scale[0]
	snode.sScale.f1 = scale[1]
	snode.sScale.f2 = scale[2]
	# not used
	snode.sScale.f3 = 0.0
	snode.sScale.f4 = 0.0
	snode.sScale.f5 = 0.0
	snode.sScale.f6 = 0.0
#	print "matrixToSNode trans(",trans[0],trans[1],trans[2],") rot=(",qrotate[0],qrotate[1],qrotate[2],qrotate[3],")"
	
	
# Clear any global variables
def clearData():
	global listOfAddedObjects
	listOfAddedObjects = {}
	
	return

# args: mapping channel entry, face data, face index, vertex index (next two will be vidx+1 and vidx+2)
def fillMappingChannelEntry(vMap,face,fidx,vidx):
	
	vMap.vVtx[vidx].x = (face.uv[0])[0]
	vMap.vVtx[vidx].y = (face.uv[0])[1]
	if len(face.uv[0]) == 3:
		vMap.vVtx[vidx].z = (face.uv[0])[2] 
	else:
		vMap.vVtx[vidx].z = 0.0
	vMap.vVtx[vidx+1].x = (face.uv[1])[0]
	vMap.vVtx[vidx+1].y = (face.uv[1])[1]
	if len(face.uv[1]) == 3:
		vMap.vVtx[vidx+1].z = (face.uv[1])[2] 
	else:
		vMap.vVtx[vidx+1].z = 0.0					
	vMap.vVtx[vidx+2].x = (face.uv[2])[0]
	vMap.vVtx[vidx+2].y = (face.uv[2])[1]
	if len(face.uv[2]) == 3:
		vMap.vVtx[vidx+2].z = (face.uv[2])[2] 
	else:
		vMap.vVtx[vidx+2].z = 0.0					
	vMap.vIdx[fidx].i[0] = vidx
	vMap.vIdx[fidx].i[1] = vidx + 1
	vMap.vIdx[fidx].i[2] = vidx + 2

# Callbacks sections and events handling

def simplecallback(event,val):
	if event == 314:
		Draw.Exit()
def mycallback(name):
	global PODfilename
	PODfilename	= name
	Draw.Register(draw_gui,input_events, gui_events)	
	return
def mycallbackCommand(evt,name):
	global export_options	
	export_options.sClPostTarget =  c_char_p(name)
	return
def mycallbackCommandArgs(evt,name):
	global export_options	
	export_options.sClPostArgs =  c_char_p(name)
	return
def mycallbackWorkingDir(evt,name):
	global export_options	
	export_options.sClPostCWD =  c_char_p(name)
	return
def input_events(evt,val):
	#print "Event: ",evt," value:",val
	return

###########################
# 	Main Event routine    #
###########################
def gui_events(evt):
	global chosen_page
	global number_of_raw
	global export_options
	global types_indices
	global types_dict
	global primitive_type_menu

	if evt == 314:


		libpathtoload = Get('scriptsdir')
			
		if libpathtoload[0] == "/":
			print "Linux"
			libpathtoload += "/libPVRGeoPOD.so."+bppversion
		else:
			print "Windows"
			libpathtoload += "\libPVRGeoPOD-"+bppversion+".dll"

		print "Loading library: ",libpathtoload
		
		pvrgp = cdll.LoadLibrary(libpathtoload)
		pvrgp.clearData()
		clearData()
		if export_options.exportMaterials == 1:
			addSceneMaterials(pvrgp)
		addSceneObjects(pvrgp)
#		print "Ambinet: x=",World.GetCurrent().getAmb()[0]," y=",World.GetCurrent().getAmb()[1]," z=",World.GetCurrent().getAmb()[2]
		pvrgp.SetUpSceneInfo(byref(SceneInfo(Get('staframe'),Get('endframe'),MyVec3f(World.GetCurrent().getAmb()[0],
		World.GetCurrent().getAmb()[1],World.GetCurrent().getAmb()[2]))))
		pvrgp.SetUpExportOptions(byref(export_options))

		showListOfAddedObjects()

#		pvrgp.showNodeSkinning(11,0)
		
		pvrgp.GeneratePODAndStoreIntoFile(c_char_p(PODfilename))
		print "Export completed"
		Draw.Exit()
	elif evt == 315:
		print "Export Canceled"
		Draw.Exit()
	elif evt == 333:
		export_options.exportGeom = 1 - export_options.exportGeom
		Draw.Redraw()
	elif evt == 334:
		export_options.exportMaterials = 1 - export_options.exportMaterials
		Draw.Redraw()
	elif evt == 335:
		export_options.bFixedPoint = 1 - export_options.bFixedPoint
		Draw.Redraw()
		
	# Tickbox of export Contoller 
	elif evt == 336:
		export_options.exportControllers = 1 - export_options.exportControllers
		chosen_page = 0		
		Draw.Redraw()
	# Static Frame number 
	elif evt == 337:
		export_options.staticFrame = static_frame_button.val
		chosen_page = 0		
		Draw.Redraw()
	
	elif evt == 338:
		export_options.cS = 1 - export_options.cS
		chosen_page = 0		
		Draw.Redraw()
	# "Normals export"
	elif evt == 339:
		export_options.exportNormals = 1 - export_options.exportNormals
		chosen_page = 0			
		Draw.Redraw()
	# Export Vertex Colours
	elif evt == 340:
		export_options.exportVertexColor = 1 - export_options.exportVertexColor
		chosen_page = 0		
		Draw.Redraw()
	# mapping channels		
	elif evt == 341:
		export_options.exportMappingChannel = 1 - export_options.exportMappingChannel
		chosen_page = 0
		Draw.Redraw()
	#	"Generate tangent-space"
	elif evt == 342:
		export_options.bTangentSpace = 1 - export_options.bTangentSpace
		chosen_page = 0
		Draw.Redraw()
	elif evt == 343:
		export_options.bInterleaved = 1 - export_options.bInterleaved
		Draw.Redraw()
	elif evt == 344:
		export_options.exportMatrices = 1 - export_options.exportMatrices
		Draw.Redraw()
	# Tickbox of Skin modifiers
	elif evt == 345:
		export_options.exportSkin = 1 - export_options.exportSkin
		chosen_page = 0		
		Draw.Redraw()
	# Static Frame number 
	elif evt == 346:
		export_options.dwBoneLimit = max_simulatnous_matrices_button.val
		chosen_page = 0		
		Draw.Redraw()
	elif evt == 347:
		export_options.ePrimType = primitive_type_menu.val - 1
		Draw.Redraw()
	elif evt == 348:
		export_options.bIndexed = 1 - export_options.bIndexed
		Draw.Redraw()
	elif evt == 349:
		export_options.eTriSort = getTriangleSortingValue()
		Draw.Redraw()
	elif evt == 350:
		export_options.bSortVtx = 1 - export_options.bSortVtx
		Draw.Redraw()

		
	# change a page with displayed vector data
	elif evt == 3151:
		chosen_page = (chosen_page+1)%( 1+number_of_raw/4 )
		Draw.Redraw()
	elif evt == ID_POSITION_EVT_TYPE:
		export_options.sVcOptPos.eType = getMenuValue(position_data_type_menu)
		Draw.Redraw()
	elif evt == ID_POSITION_EVT_A1:
		export_options.sVcOptPos.nEnable ^= DLGOPT_VECTOR_X	# if 1 then set 0 if 0 set 1
		Draw.Redraw()
	elif evt == ID_POSITION_EVT_A2:
		export_options.sVcOptPos.nEnable ^= DLGOPT_VECTOR_Y	# if 1 then set 0 if 0 set 1		
		Draw.Redraw()
	elif evt == ID_POSITION_EVT_A3:		
		export_options.sVcOptPos.nEnable ^= DLGOPT_VECTOR_Z	# if 1 then set 0 if 0 set 1		
		Draw.Redraw()
	elif evt == ID_NORMAL_EVT_TYPE:
		export_options.sVcOptNor.eType = getMenuValue(normal_data_type_menu)
		Draw.Redraw()
	elif evt == ID_NORMAL_EVT_A1:
		export_options.sVcOptNor.nEnable ^= DLGOPT_VECTOR_X	# if 1 then set 0 if 0 set 1
		Draw.Redraw()
	elif evt == ID_NORMAL_EVT_A2:
		export_options.sVcOptNor.nEnable ^= DLGOPT_VECTOR_Y	# if 1 then set 0 if 0 set 1
		Draw.Redraw()
	elif evt == ID_NORMAL_EVT_A3:
		export_options.sVcOptNor.nEnable ^= DLGOPT_VECTOR_Z	# if 1 then set 0 if 0 set 1
		Draw.Redraw()
	elif evt == ID_TANGENT_EVT_TYPE:
		export_options.sVcOptTan.eType = getMenuValue(tangent_data_type_menu)
		Draw.Redraw()
	elif evt == ID_TANGENT_EVT_A1:
		export_options.sVcOptTan.nEnable ^= DLGOPT_VECTOR_X	# if 1 then set 0 if 0 set 1
		Draw.Redraw()
	elif evt == ID_TANGENT_EVT_A2:
		export_options.sVcOptTan.nEnable ^= DLGOPT_VECTOR_Y	# if 1 then set 0 if 0 set 1
		Draw.Redraw()
	elif evt == ID_TANGENT_EVT_A3:
		export_options.sVcOptTan.nEnable ^= DLGOPT_VECTOR_Z	# if 1 then set 0 if 0 set 1
		Draw.Redraw()
	elif evt == ID_BINORMAL_EVT_TYPE:
		export_options.sVcOptBin.eType = getMenuValue(binormal_data_type_menu)
		Draw.Redraw()
	elif evt == ID_BINORMAL_EVT_A1:
		export_options.sVcOptBin.nEnable ^= DLGOPT_VECTOR_X	# if 1 then set 0 if 0 set 1
		Draw.Redraw()
	elif evt == ID_BINORMAL_EVT_A2:
		export_options.sVcOptBin.nEnable ^= DLGOPT_VECTOR_Y	# if 1 then set 0 if 0 set 1
		Draw.Redraw()
	elif evt == ID_BINORMAL_EVT_A3:
		export_options.sVcOptBin.nEnable ^= DLGOPT_VECTOR_Z	# if 1 then set 0 if 0 set 1
		Draw.Redraw()
	elif evt == ID_COLOUR_EVT_TYPE:
		export_options.sVcOptCol.eType = getMenuValue(colour_data_type_menu)
		Draw.Redraw()
	elif evt == ID_COLOUR_EVT_A1:
		export_options.sVcOptCol.nEnable ^= DLGOPT_VECTOR_X	# if 1 then set 0 if 0 set 1
		Draw.Redraw()
	elif evt == ID_COLOUR_EVT_A2:
		export_options.sVcOptCol.nEnable ^= DLGOPT_VECTOR_Y	# if 1 then set 0 if 0 set 1
		Draw.Redraw()
	elif evt == ID_COLOUR_EVT_A3:
		export_options.sVcOptCol.nEnable ^= DLGOPT_VECTOR_Z	# if 1 then set 0 if 0 set 1
		Draw.Redraw()
	elif evt == ID_COLOUR_EVT_A4:
		export_options.sVcOptCol.nEnable ^= DLGOPT_VECTOR_W	# if 1 then set 0 if 0 set 1
		Draw.Redraw()
	elif evt == ID_BONE_INDICES_EVT_TYPE:
		export_options.sVcOptBoneInd.eType = getMenuValue(bone_indices_data_type_menu)
		Draw.Redraw()									 
	elif evt == ID_BONE_INDICES_EVT_A1:
		export_options.sVcOptBoneInd.nEnable ^= DLGOPT_VECTOR_X	# if 1 then set 0 if 0 set 1
		Draw.Redraw()
	elif evt == ID_BONE_INDICES_EVT_A2:
		export_options.sVcOptBoneInd.nEnable ^= DLGOPT_VECTOR_Y	# if 1 then set 0 if 0 set 1
		Draw.Redraw()
	elif evt == ID_BONE_INDICES_EVT_A3:
		export_options.sVcOptBoneInd.nEnable ^= DLGOPT_VECTOR_Z	# if 1 then set 0 if 0 set 1
		Draw.Redraw()
	elif evt == ID_BONE_INDICES_EVT_A4:
		export_options.sVcOptBoneInd.nEnable ^= DLGOPT_VECTOR_W	# if 1 then set 0 if 0 set 1
		Draw.Redraw()
	elif evt == ID_BONE_WEIGHTS_EVT_TYPE:
		export_options.sVcOptBoneWt.eType = getMenuValue(bone_weights_data_type_menu)
		Draw.Redraw()
	elif evt == ID_BONE_WEIGHTS_EVT_A1:
		export_options.sVcOptBoneWt.nEnable ^= DLGOPT_VECTOR_X	# if 1 then set 0 if 0 set 1
		Draw.Redraw()
	elif evt == ID_BONE_WEIGHTS_EVT_A2:
		export_options.sVcOptBoneWt.nEnable ^= DLGOPT_VECTOR_Y	# if 1 then set 0 if 0 set 1
		Draw.Redraw()
	elif evt == ID_BONE_WEIGHTS_EVT_A3:
		export_options.sVcOptBoneWt.nEnable ^= DLGOPT_VECTOR_Z	# if 1 then set 0 if 0 set 1
		Draw.Redraw()
	elif evt == ID_BONE_WEIGHTS_EVT_A4:
		export_options.sVcOptBoneWt.nEnable ^= DLGOPT_VECTOR_W	# if 1 then set 0 if 0 set 1
		Draw.Redraw()
	elif evt == ID_UVW0_EVT_TYPE:
		export_options.psVcOptUVW[0].eType = getMenuValue(uvw0_data_type_menu)
		Draw.Redraw()
	elif evt == ID_UVW0_EVT_A1:
		export_options.psVcOptUVW[0].nEnable ^= DLGOPT_VECTOR_X	# if 1 then set 0 if 0 set 1
		Draw.Redraw()
	elif evt == ID_UVW0_EVT_A2:
		export_options.psVcOptUVW[0].nEnable ^= DLGOPT_VECTOR_Y	# if 1 then set 0 if 0 set 1
		Draw.Redraw()
	elif evt == ID_UVW0_EVT_A3:
		export_options.psVcOptUVW[0].nEnable ^= DLGOPT_VECTOR_Z	# if 1 then set 0 if 0 set 1
		Draw.Redraw()
	elif evt == ID_UVW1_EVT_TYPE:
		export_options.psVcOptUVW[1].eType = getMenuValue(uvw1_data_type_menu)
		Draw.Redraw()
	elif evt == ID_UVW1_EVT_A1:
		export_options.psVcOptUVW[1].nEnable ^= DLGOPT_VECTOR_X	# if 1 then set 0 if 0 set 1
		Draw.Redraw()
	elif evt == ID_UVW1_EVT_A2:
		export_options.psVcOptUVW[1].nEnable ^= DLGOPT_VECTOR_Y	# if 1 then set 0 if 0 set 1
		Draw.Redraw()
	elif evt == ID_UVW1_EVT_A3:
		export_options.psVcOptUVW[1].nEnable ^= DLGOPT_VECTOR_Z	# if 1 then set 0 if 0 set 1
		Draw.Redraw()
	elif evt == ID_UVW2_EVT_TYPE:
		export_options.psVcOptUVW[2].eType = getMenuValue(uvw2_data_type_menu)
		Draw.Redraw()
	elif evt == ID_UVW2_EVT_A1:
		export_options.psVcOptUVW[2].nEnable ^= DLGOPT_VECTOR_X	# if 1 then set 0 if 0 set 1
		Draw.Redraw()
	elif evt == ID_UVW2_EVT_A2:
		export_options.psVcOptUVW[2].nEnable ^= DLGOPT_VECTOR_Y	# if 1 then set 0 if 0 set 1
		Draw.Redraw()
	elif evt == ID_UVW2_EVT_A3:
		export_options.psVcOptUVW[2].nEnable ^= DLGOPT_VECTOR_Z	# if 1 then set 0 if 0 set 1
		Draw.Redraw()
	elif evt == ID_UVW3_EVT_TYPE:
		export_options.psVcOptUVW[3].eType = getMenuValue(uvw3_data_type_menu)
		Draw.Redraw()
	elif evt == ID_UVW3_EVT_A1:
		export_options.psVcOptUVW[3].nEnable ^= DLGOPT_VECTOR_X	# if 1 then set 0 if 0 set 1
		Draw.Redraw()
	elif evt == ID_UVW3_EVT_A2:
		export_options.psVcOptUVW[3].nEnable ^= DLGOPT_VECTOR_Y	# if 1 then set 0 if 0 set 1
		Draw.Redraw()
	elif evt == ID_UVW3_EVT_A3:
		export_options.psVcOptUVW[3].nEnable ^= DLGOPT_VECTOR_Z	# if 1 then set 0 if 0 set 1
		Draw.Redraw()
	elif evt == ID_UVW4_EVT_TYPE:
		export_options.psVcOptUVW[4].eType = getMenuValue(uvw4_data_type_menu)
		Draw.Redraw()
	elif evt == ID_UVW4_EVT_A1:
		export_options.psVcOptUVW[4].nEnable ^= DLGOPT_VECTOR_X	# if 1 then set 0 if 0 set 1
		Draw.Redraw()
	elif evt == ID_UVW4_EVT_A2:
		export_options.psVcOptUVW[4].nEnable ^= DLGOPT_VECTOR_Y	# if 1 then set 0 if 0 set 1
		Draw.Redraw()
	elif evt == ID_UVW4_EVT_A3:
		export_options.psVcOptUVW[4].nEnable ^= DLGOPT_VECTOR_Z	# if 1 then set 0 if 0 set 1
		Draw.Redraw()
	elif evt == ID_UVW5_EVT_TYPE:
		export_options.psVcOptUVW[5].eType = getMenuValue(uvw5_data_type_menu)
		Draw.Redraw()
	elif evt == ID_UVW5_EVT_A1:
		export_options.psVcOptUVW[5].nEnable ^= DLGOPT_VECTOR_X	# if 1 then set 0 if 0 set 1
		Draw.Redraw()
	elif evt == ID_UVW5_EVT_A2:
		export_options.psVcOptUVW[5].nEnable ^= DLGOPT_VECTOR_Y	# if 1 then set 0 if 0 set 1
		Draw.Redraw()
	elif evt == ID_UVW5_EVT_A3:
		export_options.psVcOptUVW[5].nEnable ^= DLGOPT_VECTOR_Z	# if 1 then set 0 if 0 set 1
		Draw.Redraw()
	elif evt == ID_UVW6_EVT_TYPE:
		export_options.psVcOptUVW[6].eType = getMenuValue(uvw6_data_type_menu)
		Draw.Redraw()
	elif evt == ID_UVW6_EVT_A1:
		export_options.psVcOptUVW[6].nEnable ^= DLGOPT_VECTOR_X	# if 1 then set 0 if 0 set 1
		Draw.Redraw()
	elif evt == ID_UVW6_EVT_A2:
		export_options.psVcOptUVW[6].nEnable ^= DLGOPT_VECTOR_Y	# if 1 then set 0 if 0 set 1
		Draw.Redraw()
	elif evt == ID_UVW6_EVT_A3:
		export_options.psVcOptUVW[6].nEnable ^= DLGOPT_VECTOR_Z	# if 1 then set 0 if 0 set 1
		Draw.Redraw()
	elif evt == ID_UVW7_EVT_TYPE:
		export_options.psVcOptUVW[7].eType = getMenuValue(uvw7_data_type_menu)
		Draw.Redraw()
	elif evt == ID_UVW7_EVT_A1:
		export_options.psVcOptUVW[7].nEnable ^= DLGOPT_VECTOR_X	# if 1 then set 0 if 0 set 1
		Draw.Redraw()
	elif evt == ID_UVW7_EVT_A2:
		export_options.psVcOptUVW[7].nEnable ^= DLGOPT_VECTOR_Y	# if 1 then set 0 if 0 set 1
		Draw.Redraw()
	elif evt == ID_UVW7_EVT_A3:
		export_options.psVcOptUVW[7].nEnable ^= DLGOPT_VECTOR_Z	# if 1 then set 0 if 0 set 1
		Draw.Redraw()

def getMenuValue(data_type_menu):
	global types_indices
	global types_dict
	for typek in types_indices:
		if types_indices[typek] == data_type_menu.val:
			for data_type in types_dict:
				if types_dict[data_type] == typek:
					return data_type
	return 0

def getTriangleSortingValue():
	global triangle_sorting_method_menu
	if triangle_sorting_method_menu.val == 1:
		return eNone
	elif triangle_sorting_method_menu.val == 2:
		return e590Blocks
	elif triangle_sorting_method_menu.val == 3:
		return ePVRTTriStrip
	return eNone

def setTriangleSortingValue():
	global export_options
	
	if export_options.eTriSort == eNone:
		return 1
	elif export_options.eTriSort == e590Blocks:
		return 2
	elif export_options.eTriSort == ePVRTTriStrip:
		return 3
	return 1

def drawLabelledGroupBox(text,textwidth,textheight,startx,starty,width,height):
	BGL.glColor3f(0.0,0.0,0.0)
	BGL.glBegin(BGL.GL_LINES)
	# first line
	BGL.glVertex2i(startx,int(starty-textheight*0.5))
	BGL.glVertex2i(startx+20,int(starty-textheight*0.5))
	# second line
	BGL.glVertex2i(startx,int(starty-textheight*0.5))
	BGL.glVertex2i(startx,int(starty-height-textheight*0.5))
	# third line
	BGL.glVertex2i(startx,int(starty-height-textheight*0.5))
	BGL.glVertex2i(startx+width,int(starty-height-textheight*0.5))
	# fourth line
	BGL.glVertex2i(startx+width,int(starty-height-textheight*0.5))
	BGL.glVertex2i(startx+width,int(starty-textheight*0.5))
	#fifth line
	BGL.glVertex2i(startx+width,int(starty-textheight*0.5))
	BGL.glVertex2i(startx+textwidth,int(starty-textheight*0.5))
	BGL.glEnd()
	# Display text
	BGL.glRasterPos2i(startx+20,starty-textheight)
	Draw.Text(text,'large')
	return				
# startx,starty is left top corner
def drawFormatControlWidget(startx,starty,name_of_format,evt_types,evt_a1,evt_a2,evt_a3,evt_a4,rtype,enabled):
	global types_indices	
	name_width =  100
	name_height = 20
	menu_width = 120
	menu_height = 20
	toggle_size = 20
	space_to_toggle = 30
	Draw.Label(name_of_format,startx,starty-name_height,name_width,name_height)
	list_of_types = "ARGB|byte| byte, normalised|D3DCOLOR|DEC3N|fixed 16.16|float|int|RGBA|short|short, normalised|unsigned byte|unsigned short"
	result = Draw.Menu(list_of_types,evt_types,startx+name_width,starty-menu_height,menu_width,menu_height,types_indices[rtype])
	Draw.Toggle(" ",evt_a1,startx+name_width+menu_width+space_to_toggle,starty-toggle_size,toggle_size,toggle_size, enabled & DLGOPT_VECTOR_X)
	Draw.Toggle(" ",evt_a2,startx+name_width+menu_width+space_to_toggle+toggle_size,starty-toggle_size,toggle_size,toggle_size,enabled & DLGOPT_VECTOR_Y)
	Draw.Toggle(" ",evt_a3,startx+name_width+menu_width+space_to_toggle+toggle_size*2,starty-toggle_size,toggle_size,toggle_size,enabled & DLGOPT_VECTOR_Z)
	if evt_a4:
		Draw.Toggle(" ",evt_a4,startx+name_width+menu_width+space_to_toggle+toggle_size*3,starty-toggle_size,toggle_size,toggle_size,enabled & DLGOPT_VECTOR_W)
	return result
				
# Draw Exports geometry options at given location
def drawExportOptions(startx,starty):
	# draw grouping Box for exports options	
	global export_options	
	global static_frame_button
	drawLabelledGroupBox(" Export options ",Draw.GetStringWidth(" Export options   ",'large'),10,startx,starty,200,450)
	step = 20
	marginx = 20
	marginy = 40
	gcb = Draw.Toggle("Geometry ",333,startx+marginx,starty-marginy,100,step,export_options.exportGeom)
	mcb = Draw.Toggle("Materials",334,startx+marginx,starty-step-marginy,100,step,export_options.exportMaterials)
#	bgb = Draw.Toggle("Bone geometry",332,startx+marginx,starty-step*2-marginy,100,step,0)
	fcb = Draw.Toggle("Fixed point",335,startx+marginx,starty-step*2-marginy,100,step,export_options.bFixedPoint)
	drawLabelledGroupBox("Animation",Draw.GetStringWidth("Animation   ",'large'),10,startx+marginx,starty-120,170,80)
	ecb = Draw.Toggle("Export controllers ",336,startx+marginx*2,starty-marginy-120,120,step,export_options.exportControllers)
	if export_options.exportControllers:
		static_frame_button = Draw.Number("Static Frame",337,startx+marginx*2,starty-marginy-120-10-step,120,step,export_options.staticFrame,0,32767,"Static Frame value")
	drawLabelledGroupBox("Coordinate System",Draw.GetStringWidth("Coordinate System   ",'large'),10,startx+marginx,starty-220,170,40)
	name = "OpenGL| DirectX"
	result = Draw.Menu(name,338,startx+marginx*2,starty-220-marginy,130,step,export_options.cS+1)
	return

def drawGeometryOptions(startx,starty):
	global export_options
	global chosen_page
	global number_of_raw
	global types_dict
	global position_data_type_menu 
	global normal_data_type_menu 
	global tangent_data_type_menu
	global binormal_data_type_menu
	global colour_data_type_menu 
	global bone_indices_data_type_menu
	global bone_weights_data_type_menu 
	global uvw0_data_type_menu 
	global uvw1_data_type_menu 
	global uvw2_data_type_menu 
	global uvw3_data_type_menu 
	global uvw4_data_type_menu 
	global uvw5_data_type_menu 
	global uvw6_data_type_menu 
	global uvw7_data_type_menu 
	global primitive_type_menu
	global triangle_sorting_method_menu
	global max_simulatnous_matrices_button
	
	step = 20
	marginx = 20
	marginy = 40
	tickboxwidth = 150
	label_width = 100
	menu_width = 120
	toggle_size = 20
	space_to_toggle = 30
	# draw grouping Box for geometry options	
	drawLabelledGroupBox(" Geometry options ",Draw.GetStringWidth(" Geometry Options   ",'large'),10,startx,starty,420,450)
	
	cb1 = Draw.Toggle("Normals",339,startx+marginx,starty - marginy,tickboxwidth,step, export_options.exportNormals)
	cb2 = Draw.Toggle("Vertex colours",340,startx+marginx,starty-step-marginy,tickboxwidth,step,export_options.exportVertexColor)
	cb3 = Draw.Toggle("Mapping channels",341,startx+marginx,starty-step*2-marginy,tickboxwidth,step,export_options.exportMappingChannel)
	cb4 = Draw.Toggle("Generate tangent-space",342,startx+marginx,starty-step*3-marginy,tickboxwidth,step,export_options.bTangentSpace)
	cb5 = Draw.Toggle("Interleave vectors",343,startx+marginx,starty-step*4-marginy,tickboxwidth,step,export_options.bInterleaved)
	cb6 = Draw.Toggle("Export Matrices",344,startx+marginx,starty-step*5-marginy,tickboxwidth,step,export_options.exportMatrices)

	# grouping box	Skin options
	drawLabelledGroupBox("Skin",Draw.GetStringWidth("Skin   ",'large'),10,startx+2*marginx+tickboxwidth,starty-step,210,80)	
	cb7 = Draw.Toggle("Export skin modifiers",345,startx+3*marginx+tickboxwidth,starty-marginy-step,tickboxwidth,step,export_options.exportSkin)
	if export_options.exportSkin:
		max_simulatnous_matrices_button = Draw.Number("Max simultanous matrices",346,startx+3*marginx+tickboxwidth,starty-marginy -2*step - 10,tickboxwidth+30,step,export_options.dwBoneLimit,0,65356,"Maximum simultanious matrices value    ")

	# grouping box	Primitive type options
	ptposy = starty-marginy*3
	drawLabelledGroupBox("Primitive type",Draw.GetStringWidth("Primitive type   ",'large'),10,startx+2*marginx+tickboxwidth,ptposy,210,70)	
	name = "Triangle list|Triangle strip"
	primitive_type_menu = Draw.Menu(name,347,startx+3*marginx+tickboxwidth,ptposy  - marginy,tickboxwidth,step,export_options.ePrimType+1)
	cb8 = Draw.Toggle("Indexed",348,startx+3*marginx+tickboxwidth,ptposy - marginy -step,tickboxwidth,step,export_options.bIndexed)

	# grouping box	Primitive type options
	ptposy = starty-marginy*5-step +10
	drawLabelledGroupBox("Triangle sorting method",Draw.GetStringWidth("Triangle sorting method   ",'large'),10,startx+2*marginx+tickboxwidth,ptposy,210,70)	
	name = "None|PVRTGeometrySort|PVRTTriStrip"
	triangle_sorting_method_menu = Draw.Menu(name,349,startx+3*marginx+tickboxwidth,ptposy  - marginy,tickboxwidth,step, setTriangleSortingValue())
	cb8 = Draw.Toggle("Sort vertices",350,startx+3*marginx+tickboxwidth,ptposy - marginy -step,tickboxwidth,step,export_options.bSortVtx)

	# grouping box Vertex vector formats
	ptposy = starty-marginy*7-step
	drawLabelledGroupBox("Vertex vector formats",Draw.GetStringWidth("Vertex vector format    ",'large'),10,startx+marginx,ptposy,tickboxwidth+230,140)	
	# Name of collumns comes below and the columns itself
	Draw.Label("Vector",startx+marginx*2,ptposy-marginy,label_width,20)
	Draw.Label("Type",startx+marginx*2+label_width,ptposy-marginy,label_width,20)
	Draw.Label("A1",startx+marginx*2+label_width+space_to_toggle+menu_width,ptposy-marginy,toggle_size,toggle_size)
	Draw.Label("A2",startx+marginx*2+label_width+space_to_toggle+menu_width+toggle_size,ptposy-marginy,toggle_size,toggle_size)
	Draw.Label("A3",startx+marginx*2+label_width+space_to_toggle+menu_width+toggle_size*2,ptposy-marginy,toggle_size,toggle_size)
	Draw.Label("A4",startx+marginx*2+label_width+space_to_toggle+menu_width+toggle_size*3,ptposy-marginy,toggle_size,toggle_size)
	
	number_of_raw = 0
	list_of_formats = ["Position", "Normal", "Tangent", "Binormal", "Colour", "Bone indices", "Bone weights", "UVW0","UVW1","UVW2","UVW3","UVW4","UVW5","UVW6","UVW7"]
	list_of_formats_descriptions = {"Position": FormatDescription(types_dict[export_options.sVcOptPos.eType],ID_POSITION_EVT_TYPE,ID_POSITION_EVT_A1,ID_POSITION_EVT_A2,ID_POSITION_EVT_A3,0,export_options.sVcOptPos.nEnable),
									 "Normal": FormatDescription(types_dict[export_options.sVcOptNor.eType],ID_NORMAL_EVT_TYPE,ID_NORMAL_EVT_A1,ID_NORMAL_EVT_A2,ID_NORMAL_EVT_A3,0,export_options.sVcOptNor.nEnable), 
									 "Tangent": FormatDescription(types_dict[export_options.sVcOptTan.eType],ID_TANGENT_EVT_TYPE,ID_TANGENT_EVT_A1,ID_TANGENT_EVT_A2,ID_TANGENT_EVT_A3,0,export_options.sVcOptTan.nEnable),
									 "Binormal": FormatDescription(types_dict[export_options.sVcOptBin.eType],ID_BINORMAL_EVT_TYPE,ID_BINORMAL_EVT_A1,ID_BINORMAL_EVT_A2,ID_BINORMAL_EVT_A3,0,export_options.sVcOptBin.nEnable),
									 "Colour": FormatDescription(types_dict[export_options.sVcOptCol.eType],ID_COLOUR_EVT_TYPE,ID_COLOUR_EVT_A1,ID_COLOUR_EVT_A2,ID_COLOUR_EVT_A3,ID_COLOUR_EVT_A4,export_options.sVcOptCol.nEnable),
									 "Bone indices": FormatDescription(types_dict[export_options.sVcOptBoneInd.eType],ID_BONE_INDICES_EVT_TYPE,ID_BONE_INDICES_EVT_A1,ID_BONE_INDICES_EVT_A2,ID_BONE_INDICES_EVT_A3,ID_BONE_INDICES_EVT_A4,export_options.sVcOptBoneInd.nEnable),
									 "Bone weights": FormatDescription(types_dict[export_options.sVcOptBoneWt.eType],ID_BONE_WEIGHTS_EVT_TYPE,ID_BONE_WEIGHTS_EVT_A1,ID_BONE_WEIGHTS_EVT_A2,ID_BONE_WEIGHTS_EVT_A3,ID_BONE_WEIGHTS_EVT_A4,export_options.sVcOptBoneWt.nEnable),
									 "UVW0": FormatDescription(types_dict[export_options.psVcOptUVW[0].eType],ID_UVW0_EVT_TYPE,ID_UVW0_EVT_A1,ID_UVW0_EVT_A2,ID_UVW0_EVT_A3,0,export_options.psVcOptUVW[0].nEnable),
									 "UVW1": FormatDescription(types_dict[export_options.psVcOptUVW[1].eType],ID_UVW1_EVT_TYPE,ID_UVW1_EVT_A1,ID_UVW1_EVT_A2,ID_UVW1_EVT_A3,0,export_options.psVcOptUVW[1].nEnable),
									 "UVW2": FormatDescription(types_dict[export_options.psVcOptUVW[2].eType],ID_UVW2_EVT_TYPE,ID_UVW2_EVT_A1,ID_UVW2_EVT_A2,ID_UVW2_EVT_A3,0,export_options.psVcOptUVW[2].nEnable),
									 "UVW3": FormatDescription(types_dict[export_options.psVcOptUVW[3].eType],ID_UVW3_EVT_TYPE,ID_UVW3_EVT_A1,ID_UVW3_EVT_A2,ID_UVW3_EVT_A3,0,export_options.psVcOptUVW[3].nEnable),
									 "UVW4": FormatDescription(types_dict[export_options.psVcOptUVW[4].eType],ID_UVW4_EVT_TYPE,ID_UVW4_EVT_A1,ID_UVW4_EVT_A2,ID_UVW4_EVT_A3,0,export_options.psVcOptUVW[4].nEnable),
									 "UVW5": FormatDescription(types_dict[export_options.psVcOptUVW[5].eType],ID_UVW5_EVT_TYPE,ID_UVW5_EVT_A1,ID_UVW5_EVT_A2,ID_UVW5_EVT_A3,0,export_options.psVcOptUVW[5].nEnable),
									 "UVW6": FormatDescription(types_dict[export_options.psVcOptUVW[6].eType],ID_UVW6_EVT_TYPE,ID_UVW6_EVT_A1,ID_UVW6_EVT_A2,ID_UVW6_EVT_A3,0,export_options.psVcOptUVW[6].nEnable),
									 "UVW7": FormatDescription(types_dict[export_options.psVcOptUVW[7].eType],ID_UVW7_EVT_TYPE,ID_UVW7_EVT_A1,ID_UVW7_EVT_A2,ID_UVW7_EVT_A3,0,export_options.psVcOptUVW[7].nEnable)}
	i = 0
	while i<len(list_of_formats):
		# warunek ze niby jak aktywne to rysuj, trza odpowiedi atrybut jeszcze gdzies dodac
		if list_of_formats[i] == "Normal" and export_options.exportNormals == 0 :
			number_of_raw+=0
		elif list_of_formats[i] == "Colour" and export_options.exportVertexColor == 0 :
			number_of_raw+=0
		elif list_of_formats[i] in ["Tangent","Binormal"] and export_options.bTangentSpace == 0 :			
			number_of_raw+=0
		elif list_of_formats[i] in ["UVW0","UVW1","UVW2","UVW3","UVW4","UVW5","UVW6","UVW7"] and export_options.exportMappingChannel == 0 :			
			number_of_raw+=0
		elif list_of_formats[i] in ["Bone indices","Bone weights"] and export_options.exportSkin == 0 :			
			number_of_raw+=0
		else:
		#	print "format: ",list_of_formats[i]," chosen page=",chosen_page," number_of_raws=",number_of_raw,"floor(numbver/4)=",floor(number_of_raw/4)," i=",i ," len of list_of_formats=",len(list_of_formats)
			if floor(number_of_raw/4) == chosen_page:
				menu_widget = drawFormatControlWidget(startx+marginx*2,ptposy-marginy-step*(number_of_raw%4),list_of_formats[i],\
				list_of_formats_descriptions[list_of_formats[i]].evt_type,list_of_formats_descriptions[list_of_formats[i]].evt1,\
				list_of_formats_descriptions[list_of_formats[i]].evt2,list_of_formats_descriptions[list_of_formats[i]].evt3,\
				list_of_formats_descriptions[list_of_formats[i]].evt4,list_of_formats_descriptions[list_of_formats[i]].type,
				list_of_formats_descriptions[list_of_formats[i]].enabled)
				if list_of_formats[i] == "Position":
					position_data_type_menu = menu_widget
				elif list_of_formats[i] == "Normal":
					normal_data_type_menu = menu_widget
				elif list_of_formats[i] == "Colour":
					colour_data_type_menu = menu_widget
				elif list_of_formats[i] == "Tangent":
					tangent_data_type_menu = menu_widget
				elif list_of_formats[i] == "Binormal":
					binormal_data_type_menu = menu_widget
				elif list_of_formats[i] == "Bone indices":
					bone_indices_data_type_menu = menu_widget
				elif list_of_formats[i] == "Bone weights":
					bone_weights_data_type_menu = menu_widget
				elif list_of_formats[i] == "UVW0":
					uvw0_data_type_menu = menu_widget
				elif list_of_formats[i] == "UVW1":
					uvw1_data_type_menu = menu_widget
				elif list_of_formats[i] == "UVW2":
					uvw2_data_type_menu = menu_widget
				elif list_of_formats[i] == "UVW3":
					uvw3_data_type_menu = menu_widget
				elif list_of_formats[i] == "UVW4":
					uvw4_data_type_menu = menu_widget
				elif list_of_formats[i] == "UVW5":
					uvw5_data_type_menu = menu_widget
				elif list_of_formats[i] == "UVW6":
					uvw6_data_type_menu = menu_widget
				elif list_of_formats[i] == "UVW7":
					uvw7_data_type_menu = menu_widget
			number_of_raw += 1
		i += 1			
	# If there are more than one page requested to show all visible raws then create a button for changing the pages
	number_of_raw-=1
	endpage = floor(number_of_raw/4)
	
#	print "chosen page=",chosen_page," number_of_raws=",number_of_raw,"floor(numbver/4)=",floor(number_of_raw/4)," i=",i ," len of list_of_formats=",len(list_of_formats)," endpage=",endpage
	
	if endpage >= 1:
		endpage+=1
		etykieta = str(int(chosen_page+1))+" of "+str(int(endpage))
		Draw.PushButton(etykieta,3151,int(startx+marginx*2+label_width+space_to_toggle+menu_width+toggle_size*2.5), int(starty - 450 + 0.5*step) ,50,15,"Change a page with displayed vector formats")
	
def drawCommandLineOptions(startx,starty):
	step = 20
	marginx = 20
	marginy = 40
	tickboxwidth = 150
	# draw grouping Box for geometry options	
	drawLabelledGroupBox(" Post export command line ",Draw.GetStringWidth(" Post export command line   ",'large'),10,startx,starty,620,80)		
	Draw.String("Command: ",351, startx + marginx, starty - marginy ,620 - marginx*2,step," ",200,"command to execute after data being exported to POD",mycallbackCommand)
	Draw.String("Command arguments: ",352, startx + marginx, starty - marginy -step ,620 - marginx*2,step," ",200,"command arguments",mycallbackCommandArgs)
	Draw.String("Working directory: ",353, startx + marginx, starty - marginy -step*2,620 - marginx*2,step," ",200,"Working directory",mycallbackWorkingDir)
	
def drawExport(startx,starty):
	global PODfilename
	step = 20
	marginx = 20
	marginy = 40
	width = 620
	drawLabelledGroupBox(" Generating POD: "+PODfilename,Draw.GetStringWidth(" Generating POD:    "+PODfilename,'large'),10,startx,starty,width,40)		
	Draw.PushButton("OK",314,startx + marginx*11, starty - marginy ,60,20,"Proceed generating of POD file")
	Draw.PushButton("Cancel",315,startx + width - marginx*11 -60, starty - marginy,60,20,"Cancel export to POD file")
	
# The function redrawing the GUI
def draw_gui():
	wymiary = Window.GetScreenSize()

	# Each element is defined by his left top corner
	# Y axe is positive in upward directon
	# X axe is positive in right directon
	# total width of GUI is 620
	drawExportOptions(wymiary[0]/2 - 310,wymiary[1]/2 + 150)
	drawGeometryOptions(wymiary[0]/2 - 110,wymiary[1]/2 + 150 )
	drawCommandLineOptions(wymiary[0]/2 - 310,wymiary[1]/2 - 320)
	drawExport(wymiary[0]/2 - 310,wymiary[1]/2 - 420)


print ">>>------"

name = "untitled.pod"

Window.FileSelector(mycallback,"OK",name)


#Window.DrawProgressBar(0.0,"Initialize Data")
#Window.DrawProgressBar(0.5,"Passing gathered data to processing library")
#Window.DrawProgressBar(1.0,"Finalizing")



print "\n---------<<<<"