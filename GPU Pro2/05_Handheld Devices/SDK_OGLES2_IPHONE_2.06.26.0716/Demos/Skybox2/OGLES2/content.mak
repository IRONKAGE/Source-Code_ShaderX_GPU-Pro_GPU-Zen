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
	Balloon.pvr \
	Balloon_pvr.pvr \
	Noise.pvr \
	Skybox.pvr \
	SkyboxMidnight.pvr

BIN_SHADERS =

RESOURCES = \
	$(CONTENTDIR)/Balloon.cpp \
	$(CONTENTDIR)/Balloon_pvr.cpp \
	$(CONTENTDIR)/Noise.cpp \
	$(CONTENTDIR)/Skybox.cpp \
	$(CONTENTDIR)/SkyboxMidnight.cpp \
	$(CONTENTDIR)/effects.cpp \
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

Balloon.pvr: $(MEDIAPATH)/balloon.png
	$(PVRTEXTOOL) -m -fOGLPVRTC4 -i$(MEDIAPATH)/balloon.png -o$@

Balloon_pvr.pvr: $(MEDIAPATH)/balloon_pvr.png
	$(PVRTEXTOOL) -m -fOGLPVRTC4 -i$(MEDIAPATH)/balloon_pvr.png -o$@

Noise.pvr: $(MEDIAPATH)/Noise.png
	$(PVRTEXTOOL) -m -fOGLPVRTC4 -i$(MEDIAPATH)/Noise.png -o$@

Skybox.pvr: $(MEDIAPATH)/skybox1.png $(MEDIAPATH)/skybox2.png $(MEDIAPATH)/skybox3.png $(MEDIAPATH)/skybox4.png $(MEDIAPATH)/skybox5.png $(MEDIAPATH)/skybox6.png
	$(PVRTEXTOOL) -s -m -p -fOGLPVRTC4 -i$(MEDIAPATH)/skybox1.png -o$@

SkyboxMidnight.pvr: $(MEDIAPATH)/skybox_midnight1.png $(MEDIAPATH)/skybox_midnight2.png $(MEDIAPATH)/skybox_midnight3.png $(MEDIAPATH)/skybox_midnight4.png $(MEDIAPATH)/skybox_midnight5.png $(MEDIAPATH)/skybox_midnight6.png
	$(PVRTEXTOOL) -s -m -p -fOGLPVRTC4 -i$(MEDIAPATH)/skybox_midnight1.png -o$@

$(CONTENTDIR)/Balloon.cpp: Balloon.pvr
	$(FILEWRAP)  -o $@ Balloon.pvr

$(CONTENTDIR)/Balloon_pvr.cpp: Balloon_pvr.pvr
	$(FILEWRAP)  -o $@ Balloon_pvr.pvr

$(CONTENTDIR)/Noise.cpp: Noise.pvr
	$(FILEWRAP)  -o $@ Noise.pvr

$(CONTENTDIR)/Skybox.cpp: Skybox.pvr
	$(FILEWRAP)  -o $@ Skybox.pvr

$(CONTENTDIR)/SkyboxMidnight.cpp: SkyboxMidnight.pvr
	$(FILEWRAP)  -o $@ SkyboxMidnight.pvr

$(CONTENTDIR)/effects.cpp: effects.pfx
	$(FILEWRAP)  -s  -o $@ effects.pfx

$(CONTENTDIR)/Scene.cpp: Scene.pod
	$(FILEWRAP)  -o $@ Scene.pod

############################################################################
# End of file (content.mak)
############################################################################
