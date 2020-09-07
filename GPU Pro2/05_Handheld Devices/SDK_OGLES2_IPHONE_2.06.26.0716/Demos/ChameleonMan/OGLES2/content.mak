#--------------------------------------------------------------------------
# Name         : content.mak
# Title        : Makefile to build content files
# Author       : Auto-generated
# Created      : 22/02/2010
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
# $Revision: 1.1 $
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
	skyline.pvr \
	Wall_diffuse_baked.pvr \
	Tang_space_BodyMap.pvr \
	Tang_space_LegsMap.pvr \
	Tang_space_BeltMap.pvr \
	FinalChameleonManLegs.pvr \
	FinalChameleonManHeadBody.pvr \
	lamp.pvr \
	ChameleonBelt.pvr

BIN_SHADERS = \
	SkinnedVertShader.vsc \
	SkinnedFragShader.fsc \
	DefaultVertShader.vsc \
	DefaultFragShader.fsc

RESOURCES = \
	$(CONTENTDIR)/ChameleonScene.cpp \
	$(CONTENTDIR)/skyline.cpp \
	$(CONTENTDIR)/Wall_diffuse_baked.cpp \
	$(CONTENTDIR)/Tang_space_BodyMap.cpp \
	$(CONTENTDIR)/Tang_space_LegsMap.cpp \
	$(CONTENTDIR)/Tang_space_BeltMap.cpp \
	$(CONTENTDIR)/FinalChameleonManLegs.cpp \
	$(CONTENTDIR)/FinalChameleonManHeadBody.cpp \
	$(CONTENTDIR)/lamp.cpp \
	$(CONTENTDIR)/ChameleonBelt.cpp \
	$(CONTENTDIR)/SkinnedVertShader.cpp \
	$(CONTENTDIR)/SkinnedFragShader.cpp \
	$(CONTENTDIR)/DefaultVertShader.cpp \
	$(CONTENTDIR)/DefaultFragShader.cpp

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

skyline.pvr: $(MEDIAPATH)/skyline.png
	$(PVRTEXTOOL) -m -fOGLPVRTC4 -i$(MEDIAPATH)/skyline.png -o$@

Wall_diffuse_baked.pvr: $(MEDIAPATH)/Wall_diffuse_baked.png
	$(PVRTEXTOOL) -m -fOGLPVRTC4 -i$(MEDIAPATH)/Wall_diffuse_baked.png -o$@

Tang_space_BodyMap.pvr: $(MEDIAPATH)/Tang_space_BodyMap.png
	$(PVRTEXTOOL) -fOGLPVRTC4 -i$(MEDIAPATH)/Tang_space_BodyMap.png -o$@

Tang_space_LegsMap.pvr: $(MEDIAPATH)/Tang_space_LegsMap.png
	$(PVRTEXTOOL) -fOGLPVRTC4 -i$(MEDIAPATH)/Tang_space_LegsMap.png -o$@

Tang_space_BeltMap.pvr: $(MEDIAPATH)/Tang_space_BeltMap.png
	$(PVRTEXTOOL) -fOGLPVRTC4 -i$(MEDIAPATH)/Tang_space_BeltMap.png -o$@

FinalChameleonManLegs.pvr: $(MEDIAPATH)/FinalChameleonManLegs.png
	$(PVRTEXTOOL) -m -fOGLPVRTC4 -i$(MEDIAPATH)/FinalChameleonManLegs.png -o$@

FinalChameleonManHeadBody.pvr: $(MEDIAPATH)/FinalChameleonManHeadBody.png
	$(PVRTEXTOOL) -m -fOGLPVRTC4 -i$(MEDIAPATH)/FinalChameleonManHeadBody.png -o$@

lamp.pvr: $(MEDIAPATH)/lamp.png
	$(PVRTEXTOOL) -m -fOGLPVRTC2 -i$(MEDIAPATH)/lamp.png -o$@

ChameleonBelt.pvr: $(MEDIAPATH)/ChameleonBelt.png
	$(PVRTEXTOOL) -m -fOGLPVRTC2 -i$(MEDIAPATH)/ChameleonBelt.png -o$@

$(CONTENTDIR)/ChameleonScene.cpp: ChameleonScene.pod
	$(FILEWRAP)  -o $@ ChameleonScene.pod

$(CONTENTDIR)/skyline.cpp: skyline.pvr
	$(FILEWRAP)  -o $@ skyline.pvr

$(CONTENTDIR)/Wall_diffuse_baked.cpp: Wall_diffuse_baked.pvr
	$(FILEWRAP)  -o $@ Wall_diffuse_baked.pvr

$(CONTENTDIR)/Tang_space_BodyMap.cpp: Tang_space_BodyMap.pvr
	$(FILEWRAP)  -o $@ Tang_space_BodyMap.pvr

$(CONTENTDIR)/Tang_space_LegsMap.cpp: Tang_space_LegsMap.pvr
	$(FILEWRAP)  -o $@ Tang_space_LegsMap.pvr

$(CONTENTDIR)/Tang_space_BeltMap.cpp: Tang_space_BeltMap.pvr
	$(FILEWRAP)  -o $@ Tang_space_BeltMap.pvr

$(CONTENTDIR)/FinalChameleonManLegs.cpp: FinalChameleonManLegs.pvr
	$(FILEWRAP)  -o $@ FinalChameleonManLegs.pvr

$(CONTENTDIR)/FinalChameleonManHeadBody.cpp: FinalChameleonManHeadBody.pvr
	$(FILEWRAP)  -o $@ FinalChameleonManHeadBody.pvr

$(CONTENTDIR)/lamp.cpp: lamp.pvr
	$(FILEWRAP)  -o $@ lamp.pvr

$(CONTENTDIR)/ChameleonBelt.cpp: ChameleonBelt.pvr
	$(FILEWRAP)  -o $@ ChameleonBelt.pvr

$(CONTENTDIR)/SkinnedVertShader.cpp: SkinnedVertShader.vsh SkinnedVertShader.vsc
	$(FILEWRAP)  -s  -o $@ SkinnedVertShader.vsh
	$(FILEWRAP)  -oa $@ SkinnedVertShader.vsc

$(CONTENTDIR)/SkinnedFragShader.cpp: SkinnedFragShader.fsh SkinnedFragShader.fsc
	$(FILEWRAP)  -s  -o $@ SkinnedFragShader.fsh
	$(FILEWRAP)  -oa $@ SkinnedFragShader.fsc

$(CONTENTDIR)/DefaultVertShader.cpp: DefaultVertShader.vsh DefaultVertShader.vsc
	$(FILEWRAP)  -s  -o $@ DefaultVertShader.vsh
	$(FILEWRAP)  -oa $@ DefaultVertShader.vsc

$(CONTENTDIR)/DefaultFragShader.cpp: DefaultFragShader.fsh DefaultFragShader.fsc
	$(FILEWRAP)  -s  -o $@ DefaultFragShader.fsh
	$(FILEWRAP)  -oa $@ DefaultFragShader.fsc

SkinnedVertShader.vsc: SkinnedVertShader.vsh
	$(PVRUNISCO) SkinnedVertShader.vsh $@  -v 

SkinnedFragShader.fsc: SkinnedFragShader.fsh
	$(PVRUNISCO) SkinnedFragShader.fsh $@  -f 

DefaultVertShader.vsc: DefaultVertShader.vsh
	$(PVRUNISCO) DefaultVertShader.vsh $@  -v 

DefaultFragShader.fsc: DefaultFragShader.fsh
	$(PVRUNISCO) DefaultFragShader.fsh $@  -f 

############################################################################
# End of file (content.mak)
############################################################################
