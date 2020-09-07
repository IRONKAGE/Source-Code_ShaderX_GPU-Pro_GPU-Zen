#--------------------------------------------------------------------------
# Name         : content.mak
# Title        : Makefile to build content files
# Author       : Auto-generated
# Created      : 10/12/2009
#
# Copyright    : 2007 by Imagination Technologies.  All rights reserved.
#              : No part of this software, either material or conceptual 
#              : may be copied or distributed, transmitted, transcribed,
#              : stored in a retrieval system or translated into any 
#              : human or computer language in any form by any means,
#              : electronic, mechanical, manual or other-wise, or 
#              : disclosed to third parties without the express written
#              : permission of VideoLogic Limited, Unit 8, HomePark
#              : Industrial Estate, King's Langley, Hertfordshire,
#              : WD4 8LZ, U.K.
#
# Description  : Makefile to build content files for demos in the PowerVR SDK
#
# Platform     :
#
# $Revision: 1.9 $
#--------------------------------------------------------------------------

#############################################################################
## Variables
#############################################################################

PVRTEXTOOL 	= ..\..\..\Utilities\PVRTexTool\PVRTexToolCL\Win32\PVRTexTool.exe
FILEWRAP 	= ..\..\..\Utilities\Filewrap\Win32\Filewrap.exe
PVRUNISCO 	= ..\..\..\Utilities\PVRUniSCo\OGLES\Win32\PVRUniSCo.exe

MEDIAPATH = ../Media
CONTENTDIR = Content

#############################################################################
## Instructions
#############################################################################

TEXTURES = \
	NewNormalMap.pvr \
	Mountain.pvr \
	wood.pvr \
	sail.pvr \
	MountainFloor.pvr

BIN_SHADERS = \
	FragShader.fsc \
	VertShader.vsc \
	SkyboxFShader.fsc \
	SkyboxVShader.vsc \
	ModelFShader.fsc \
	ModelVShader.vsc \
	Tex2DFShader.fsc \
	Tex2DVShader.vsc \
	PlaneTexFShader.fsc \
	PlaneTexVShader.vsc

RESOURCES = \
	$(CONTENTDIR)/NewNormalMap.cpp \
	$(CONTENTDIR)/Mountain.cpp \
	$(CONTENTDIR)/wood.cpp \
	$(CONTENTDIR)/sail.cpp \
	$(CONTENTDIR)/MountainFloor.cpp \
	$(CONTENTDIR)/FragShader.cpp \
	$(CONTENTDIR)/VertShader.cpp \
	$(CONTENTDIR)/SkyboxFShader.cpp \
	$(CONTENTDIR)/SkyboxVShader.cpp \
	$(CONTENTDIR)/ModelFShader.cpp \
	$(CONTENTDIR)/ModelVShader.cpp \
	$(CONTENTDIR)/Tex2DFShader.cpp \
	$(CONTENTDIR)/Tex2DVShader.cpp \
	$(CONTENTDIR)/PlaneTexFShader.cpp \
	$(CONTENTDIR)/PlaneTexVShader.cpp \
	$(CONTENTDIR)/Scene.cpp

all: resources
	
help:
	@echo Valid targets are:
	@echo resources, textures, binary_shaders, clean
	@echo PVRTEXTOOL, FILEWRAP and PVRUNISCO can be used to override the default paths to these utilities.

clean:
	-rm $(RESOURCES)
	-rm $(BIN_SHADERS)
	-rm $(TEXTURES)

resources: 		$(CONTENTDIR) $(RESOURCES)
textures: 		$(TEXTURES)
binary_shaders:	$(BIN_SHADERS)

$(CONTENTDIR):
	-mkdir $@

NewNormalMap.pvr: $(MEDIAPATH)/NewNormalMap.png
	$(PVRTEXTOOL) -b1.0 -m -fOGLPVRTC2 -i$(MEDIAPATH)/NewNormalMap.png -o$@

Mountain.pvr: $(MEDIAPATH)/mountain1.png $(MEDIAPATH)/mountain2.png $(MEDIAPATH)/mountain3.png $(MEDIAPATH)/mountain4.png $(MEDIAPATH)/mountain5.png $(MEDIAPATH)/mountain6.png
	$(PVRTEXTOOL) -s -m -p -fOGLPVRTC4 -i$(MEDIAPATH)/mountain1.png -o$@

wood.pvr: $(MEDIAPATH)/wood.png
	$(PVRTEXTOOL) -m -fOGLPVRTC2 -i$(MEDIAPATH)/wood.png -o$@

sail.pvr: $(MEDIAPATH)/sail.png
	$(PVRTEXTOOL) -m -fOGLPVRTC2 -i$(MEDIAPATH)/sail.png -o$@

MountainFloor.pvr: $(MEDIAPATH)/mountain6.png
	$(PVRTEXTOOL) -m -fOGLPVRTC2 -i$(MEDIAPATH)/mountain6.png -o$@

$(CONTENTDIR)/NewNormalMap.cpp: NewNormalMap.pvr
	$(FILEWRAP)  -o $@ NewNormalMap.pvr

$(CONTENTDIR)/Mountain.cpp: Mountain.pvr
	$(FILEWRAP)  -o $@ Mountain.pvr

$(CONTENTDIR)/wood.cpp: wood.pvr
	$(FILEWRAP)  -o $@ wood.pvr

$(CONTENTDIR)/sail.cpp: sail.pvr
	$(FILEWRAP)  -o $@ sail.pvr

$(CONTENTDIR)/MountainFloor.cpp: MountainFloor.pvr
	$(FILEWRAP)  -o $@ MountainFloor.pvr

$(CONTENTDIR)/FragShader.cpp: FragShader.fsh FragShader.fsc
	$(FILEWRAP)  -s  -o $@ FragShader.fsh
	$(FILEWRAP)  -oa $@ FragShader.fsc

$(CONTENTDIR)/VertShader.cpp: VertShader.vsh VertShader.vsc
	$(FILEWRAP)  -s  -o $@ VertShader.vsh
	$(FILEWRAP)  -oa $@ VertShader.vsc

$(CONTENTDIR)/SkyboxFShader.cpp: SkyboxFShader.fsh SkyboxFShader.fsc
	$(FILEWRAP)  -s  -o $@ SkyboxFShader.fsh
	$(FILEWRAP)  -oa $@ SkyboxFShader.fsc

$(CONTENTDIR)/SkyboxVShader.cpp: SkyboxVShader.vsh SkyboxVShader.vsc
	$(FILEWRAP)  -s  -o $@ SkyboxVShader.vsh
	$(FILEWRAP)  -oa $@ SkyboxVShader.vsc

$(CONTENTDIR)/ModelFShader.cpp: ModelFShader.fsh ModelFShader.fsc
	$(FILEWRAP)  -s  -o $@ ModelFShader.fsh
	$(FILEWRAP)  -oa $@ ModelFShader.fsc

$(CONTENTDIR)/ModelVShader.cpp: ModelVShader.vsh ModelVShader.vsc
	$(FILEWRAP)  -s  -o $@ ModelVShader.vsh
	$(FILEWRAP)  -oa $@ ModelVShader.vsc

$(CONTENTDIR)/Tex2DFShader.cpp: Tex2DFShader.fsh Tex2DFShader.fsc
	$(FILEWRAP)  -s  -o $@ Tex2DFShader.fsh
	$(FILEWRAP)  -oa $@ Tex2DFShader.fsc

$(CONTENTDIR)/Tex2DVShader.cpp: Tex2DVShader.vsh Tex2DVShader.vsc
	$(FILEWRAP)  -s  -o $@ Tex2DVShader.vsh
	$(FILEWRAP)  -oa $@ Tex2DVShader.vsc

$(CONTENTDIR)/PlaneTexFShader.cpp: PlaneTexFShader.fsh PlaneTexFShader.fsc
	$(FILEWRAP)  -s  -o $@ PlaneTexFShader.fsh
	$(FILEWRAP)  -oa $@ PlaneTexFShader.fsc

$(CONTENTDIR)/PlaneTexVShader.cpp: PlaneTexVShader.vsh PlaneTexVShader.vsc
	$(FILEWRAP)  -s  -o $@ PlaneTexVShader.vsh
	$(FILEWRAP)  -oa $@ PlaneTexVShader.vsc

$(CONTENTDIR)/Scene.cpp: Scene.pod
	$(FILEWRAP)  -o $@ Scene.pod

FragShader.fsc: FragShader.fsh
	$(PVRUNISCO) FragShader.fsh $@  -f 

VertShader.vsc: VertShader.vsh
	$(PVRUNISCO) VertShader.vsh $@  -v 

SkyboxFShader.fsc: SkyboxFShader.fsh
	$(PVRUNISCO) SkyboxFShader.fsh $@  -f 

SkyboxVShader.vsc: SkyboxVShader.vsh
	$(PVRUNISCO) SkyboxVShader.vsh $@  -v 

ModelFShader.fsc: ModelFShader.fsh
	$(PVRUNISCO) ModelFShader.fsh $@  -f 

ModelVShader.vsc: ModelVShader.vsh
	$(PVRUNISCO) ModelVShader.vsh $@  -v 

Tex2DFShader.fsc: Tex2DFShader.fsh
	$(PVRUNISCO) Tex2DFShader.fsh $@  -f 

Tex2DVShader.vsc: Tex2DVShader.vsh
	$(PVRUNISCO) Tex2DVShader.vsh $@  -v 

PlaneTexFShader.fsc: PlaneTexFShader.fsh
	$(PVRUNISCO) PlaneTexFShader.fsh $@  -f 

PlaneTexVShader.vsc: PlaneTexVShader.vsh
	$(PVRUNISCO) PlaneTexVShader.vsh $@  -v 

############################################################################
# End of file (content.mak)
############################################################################
