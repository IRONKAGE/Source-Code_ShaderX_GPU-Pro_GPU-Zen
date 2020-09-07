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
	MaskMain.pvr \
	RoomStill.pvr

BIN_SHADERS = \
	SHVertShader.vsc \
	DiffuseVertShader.vsc \
	FragShader.fsc

RESOURCES = \
	$(CONTENTDIR)/PhantomMask.cpp \
	$(CONTENTDIR)/MaskMain.cpp \
	$(CONTENTDIR)/RoomStill.cpp \
	$(CONTENTDIR)/SHVertShader.cpp \
	$(CONTENTDIR)/DiffuseVertShader.cpp \
	$(CONTENTDIR)/FragShader.cpp

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

MaskMain.pvr: $(MEDIAPATH)/MaskMain.png
	$(PVRTEXTOOL) -m -fOGLPVRTC4 -i$(MEDIAPATH)/MaskMain.png -o$@

RoomStill.pvr: $(MEDIAPATH)/RoomStill.png
	$(PVRTEXTOOL) -m -fOGLPVRTC4 -i$(MEDIAPATH)/RoomStill.png -o$@

$(CONTENTDIR)/PhantomMask.cpp: PhantomMask.pod
	$(FILEWRAP)  -o $@ PhantomMask.pod

$(CONTENTDIR)/MaskMain.cpp: MaskMain.pvr
	$(FILEWRAP)  -o $@ MaskMain.pvr

$(CONTENTDIR)/RoomStill.cpp: RoomStill.pvr
	$(FILEWRAP)  -o $@ RoomStill.pvr

$(CONTENTDIR)/SHVertShader.cpp: SHVertShader.vsh SHVertShader.vsc
	$(FILEWRAP)  -s  -o $@ SHVertShader.vsh
	$(FILEWRAP)  -oa $@ SHVertShader.vsc

$(CONTENTDIR)/DiffuseVertShader.cpp: DiffuseVertShader.vsh DiffuseVertShader.vsc
	$(FILEWRAP)  -s  -o $@ DiffuseVertShader.vsh
	$(FILEWRAP)  -oa $@ DiffuseVertShader.vsc

$(CONTENTDIR)/FragShader.cpp: FragShader.fsh FragShader.fsc
	$(FILEWRAP)  -s  -o $@ FragShader.fsh
	$(FILEWRAP)  -oa $@ FragShader.fsc

SHVertShader.vsc: SHVertShader.vsh
	$(PVRUNISCO) SHVertShader.vsh $@  -v 

DiffuseVertShader.vsc: DiffuseVertShader.vsh
	$(PVRUNISCO) DiffuseVertShader.vsh $@  -v 

FragShader.fsc: FragShader.fsh
	$(PVRUNISCO) FragShader.fsh $@  -f 

############################################################################
# End of file (content.mak)
############################################################################
