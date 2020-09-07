#--------------------------------------------------------------------------
# Name         : content.mak
# Title        : Makefile to build content files
# Author       : Auto-generated
# Created      : 10/09/2009
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
# $Revision: 1.3 $
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
	Table.pvr \
	Floor.pvr \
	Wall.pvr \
	TV.pvr \
	TVCase.pvr \
	TVSpeaker.pvr \
	Alum.pvr \
	Skirting.pvr \
	Camera.pvr

BIN_SHADERS = \
	FragShader.fsc \
	BWFragShader.fsc \
	VertShader.vsc

RESOURCES = \
	$(CONTENTDIR)/Table.cpp \
	$(CONTENTDIR)/Floor.cpp \
	$(CONTENTDIR)/Wall.cpp \
	$(CONTENTDIR)/TV.cpp \
	$(CONTENTDIR)/TVCase.cpp \
	$(CONTENTDIR)/TVSpeaker.cpp \
	$(CONTENTDIR)/Alum.cpp \
	$(CONTENTDIR)/Skirting.cpp \
	$(CONTENTDIR)/Camera.cpp \
	$(CONTENTDIR)/FragShader.cpp \
	$(CONTENTDIR)/BWFragShader.cpp \
	$(CONTENTDIR)/VertShader.cpp \
	$(CONTENTDIR)/FilmTVScene.cpp

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

Table.pvr: $(MEDIAPATH)/Table.png
	$(PVRTEXTOOL) -m -fOGLPVRTC4 -i$(MEDIAPATH)/Table.png -o$@

Floor.pvr: $(MEDIAPATH)/Floor.png
	$(PVRTEXTOOL) -m -fOGLPVRTC4 -i$(MEDIAPATH)/Floor.png -o$@

Wall.pvr: $(MEDIAPATH)/Wall.png
	$(PVRTEXTOOL) -m -fOGLPVRTC4 -i$(MEDIAPATH)/Wall.png -o$@

TV.pvr: $(MEDIAPATH)/TV.png
	$(PVRTEXTOOL) -m -fOGLPVRTC4 -i$(MEDIAPATH)/TV.png -o$@

TVCase.pvr: $(MEDIAPATH)/TVCase.png
	$(PVRTEXTOOL) -m -fOGLPVRTC4 -i$(MEDIAPATH)/TVCase.png -o$@

TVSpeaker.pvr: $(MEDIAPATH)/TVSpeaker.png
	$(PVRTEXTOOL) -m -fOGLPVRTC4 -i$(MEDIAPATH)/TVSpeaker.png -o$@

Alum.pvr: $(MEDIAPATH)/Alum.png
	$(PVRTEXTOOL) -m -fOGLPVRTC4 -i$(MEDIAPATH)/Alum.png -o$@

Skirting.pvr: $(MEDIAPATH)/Skirting.png
	$(PVRTEXTOOL) -m -fOGLPVRTC4 -i$(MEDIAPATH)/Skirting.png -o$@

Camera.pvr: $(MEDIAPATH)/Camera.png
	$(PVRTEXTOOL) -m -fOGLPVRTC4 -i$(MEDIAPATH)/Camera.png -o$@

$(CONTENTDIR)/Table.cpp: Table.pvr
	$(FILEWRAP)  -o $@ Table.pvr

$(CONTENTDIR)/Floor.cpp: Floor.pvr
	$(FILEWRAP)  -o $@ Floor.pvr

$(CONTENTDIR)/Wall.cpp: Wall.pvr
	$(FILEWRAP)  -o $@ Wall.pvr

$(CONTENTDIR)/TV.cpp: TV.pvr
	$(FILEWRAP)  -o $@ TV.pvr

$(CONTENTDIR)/TVCase.cpp: TVCase.pvr
	$(FILEWRAP)  -o $@ TVCase.pvr

$(CONTENTDIR)/TVSpeaker.cpp: TVSpeaker.pvr
	$(FILEWRAP)  -o $@ TVSpeaker.pvr

$(CONTENTDIR)/Alum.cpp: Alum.pvr
	$(FILEWRAP)  -o $@ Alum.pvr

$(CONTENTDIR)/Skirting.cpp: Skirting.pvr
	$(FILEWRAP)  -o $@ Skirting.pvr

$(CONTENTDIR)/Camera.cpp: Camera.pvr
	$(FILEWRAP)  -o $@ Camera.pvr

$(CONTENTDIR)/FragShader.cpp: FragShader.fsh FragShader.fsc
	$(FILEWRAP)  -s  -o $@ FragShader.fsh
	$(FILEWRAP)  -oa $@ FragShader.fsc

$(CONTENTDIR)/BWFragShader.cpp: BWFragShader.fsh BWFragShader.fsc
	$(FILEWRAP)  -s  -o $@ BWFragShader.fsh
	$(FILEWRAP)  -oa $@ BWFragShader.fsc

$(CONTENTDIR)/VertShader.cpp: VertShader.vsh VertShader.vsc
	$(FILEWRAP)  -s  -o $@ VertShader.vsh
	$(FILEWRAP)  -oa $@ VertShader.vsc

$(CONTENTDIR)/FilmTVScene.cpp: FilmTVScene.pod
	$(FILEWRAP)  -o $@ FilmTVScene.pod

FragShader.fsc: FragShader.fsh
	$(PVRUNISCO) FragShader.fsh $@  -f 

BWFragShader.fsc: BWFragShader.fsh
	$(PVRUNISCO) BWFragShader.fsh $@  -f 

VertShader.vsc: VertShader.vsh
	$(PVRUNISCO) VertShader.vsh $@  -v 

############################################################################
# End of file (content.mak)
############################################################################
