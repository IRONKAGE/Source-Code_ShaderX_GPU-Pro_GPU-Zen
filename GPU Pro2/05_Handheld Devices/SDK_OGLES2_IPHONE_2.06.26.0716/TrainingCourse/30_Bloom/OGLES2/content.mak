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
# $Revision: 1.2 $
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
	BaseTex.pvr \
	bloom_mapping.pvr

BIN_SHADERS = \
	PostBloomFragShader.fsc \
	PostBloomVertShader.vsc \
	PreBloomFragShader.fsc \
	PreBloomVertShader.vsc \
	BlurFragShader.fsc \
	BlurVertShader.vsc \
	FragShader.fsc \
	VertShader.vsc

RESOURCES = \
	$(CONTENTDIR)/BaseTex.cpp \
	$(CONTENTDIR)/bloom_mapping.cpp \
	$(CONTENTDIR)/PostBloomFragShader.cpp \
	$(CONTENTDIR)/PostBloomVertShader.cpp \
	$(CONTENTDIR)/PreBloomFragShader.cpp \
	$(CONTENTDIR)/PreBloomVertShader.cpp \
	$(CONTENTDIR)/BlurFragShader.cpp \
	$(CONTENTDIR)/BlurVertShader.cpp \
	$(CONTENTDIR)/FragShader.cpp \
	$(CONTENTDIR)/VertShader.cpp \
	$(CONTENTDIR)/Mask.cpp

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

BaseTex.pvr: $(MEDIAPATH)/tex_base.png
	$(PVRTEXTOOL) -m -fOGLPVRTC4 -i$(MEDIAPATH)/tex_base.png -o$@

bloom_mapping.pvr: $(MEDIAPATH)/bloom_mapping.png
	$(PVRTEXTOOL) -m -fOGLPVRTC4 -i$(MEDIAPATH)/bloom_mapping.png -o$@

$(CONTENTDIR)/BaseTex.cpp: BaseTex.pvr
	$(FILEWRAP)  -o $@ BaseTex.pvr

$(CONTENTDIR)/bloom_mapping.cpp: bloom_mapping.pvr
	$(FILEWRAP)  -o $@ bloom_mapping.pvr

$(CONTENTDIR)/PostBloomFragShader.cpp: PostBloomFragShader.fsh PostBloomFragShader.fsc
	$(FILEWRAP)  -s  -o $@ PostBloomFragShader.fsh
	$(FILEWRAP)  -oa $@ PostBloomFragShader.fsc

$(CONTENTDIR)/PostBloomVertShader.cpp: PostBloomVertShader.vsh PostBloomVertShader.vsc
	$(FILEWRAP)  -s  -o $@ PostBloomVertShader.vsh
	$(FILEWRAP)  -oa $@ PostBloomVertShader.vsc

$(CONTENTDIR)/PreBloomFragShader.cpp: PreBloomFragShader.fsh PreBloomFragShader.fsc
	$(FILEWRAP)  -s  -o $@ PreBloomFragShader.fsh
	$(FILEWRAP)  -oa $@ PreBloomFragShader.fsc

$(CONTENTDIR)/PreBloomVertShader.cpp: PreBloomVertShader.vsh PreBloomVertShader.vsc
	$(FILEWRAP)  -s  -o $@ PreBloomVertShader.vsh
	$(FILEWRAP)  -oa $@ PreBloomVertShader.vsc

$(CONTENTDIR)/BlurFragShader.cpp: BlurFragShader.fsh BlurFragShader.fsc
	$(FILEWRAP)  -s  -o $@ BlurFragShader.fsh
	$(FILEWRAP)  -oa $@ BlurFragShader.fsc

$(CONTENTDIR)/BlurVertShader.cpp: BlurVertShader.vsh BlurVertShader.vsc
	$(FILEWRAP)  -s  -o $@ BlurVertShader.vsh
	$(FILEWRAP)  -oa $@ BlurVertShader.vsc

$(CONTENTDIR)/FragShader.cpp: FragShader.fsh FragShader.fsc
	$(FILEWRAP)  -s  -o $@ FragShader.fsh
	$(FILEWRAP)  -oa $@ FragShader.fsc

$(CONTENTDIR)/VertShader.cpp: VertShader.vsh VertShader.vsc
	$(FILEWRAP)  -s  -o $@ VertShader.vsh
	$(FILEWRAP)  -oa $@ VertShader.vsc

$(CONTENTDIR)/Mask.cpp: Mask.pod
	$(FILEWRAP)  -o $@ Mask.pod

PostBloomFragShader.fsc: PostBloomFragShader.fsh
	$(PVRUNISCO) PostBloomFragShader.fsh $@  -f 

PostBloomVertShader.vsc: PostBloomVertShader.vsh
	$(PVRUNISCO) PostBloomVertShader.vsh $@  -v 

PreBloomFragShader.fsc: PreBloomFragShader.fsh
	$(PVRUNISCO) PreBloomFragShader.fsh $@  -f 

PreBloomVertShader.vsc: PreBloomVertShader.vsh
	$(PVRUNISCO) PreBloomVertShader.vsh $@  -v 

BlurFragShader.fsc: BlurFragShader.fsh
	$(PVRUNISCO) BlurFragShader.fsh $@  -f 

BlurVertShader.vsc: BlurVertShader.vsh
	$(PVRUNISCO) BlurVertShader.vsh $@  -v 

FragShader.fsc: FragShader.fsh
	$(PVRUNISCO) FragShader.fsh $@  -f 

VertShader.vsc: VertShader.vsh
	$(PVRUNISCO) VertShader.vsh $@  -v 

############################################################################
# End of file (content.mak)
############################################################################
