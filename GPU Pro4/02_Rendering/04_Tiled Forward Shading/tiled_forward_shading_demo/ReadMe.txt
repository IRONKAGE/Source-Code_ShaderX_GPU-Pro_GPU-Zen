Demo code implementing Tiled Forward Shading, accompanying the article 'Tiled Forward Shading'
in GPU Pro 4, Taylor and Francis 2013.

BibTex entry:
---------------
?

Overview
-----------
The demo implements Tiled Forward Shading supporting transparent geometry 
(through good old alpha blending). Also implemented is Tiled Deferred Shading,
which doesn't. 


Hardware Requirements
-----------------------
The demo should run on any graphics card supporting OpenGL 3.3. However, the
optimized CUDA implementation of Tiled Deferred requires an NVIDIA graphics card
with FERMI architecture or higher. The CUDA code should support older hardware, 
but the G-Buffer repackaging requires imageStore, which is DX11 anyway.

If CUDA is not avalable, the CUDA module is not instantiated and the glsl based 
version is used. All CUDA code can be disabled using the flag
ENABLE_CUDA_OPT_TILED_DEFERRED in Config.h.


Usage Instructions
---------------------
Run the executable by starting the file 'run.cmd'. If this fails, the most
probable reason is that the system doesn't have the Visual Studio 
redistributable package(s) installed. Run 'install_redist.cmd' to install
the redistributables for Visual C++ 2008 and 2005, both are required as
the DevIL binaries are linked against the 2005 version.

By default the demo will load the scene 'data/crysponza_bubbles/sponza.obj', 
which is a modified and repackaged version of the scene made available by 
Crytek:
http://www.crytek.com/cryengine/cryengine3/downloads

To use another scene, just replace the command line argument. The only 
supported scene format is obj.

When the demo is running, pressing <F1> brings up an info/help text which 
provides details on other function keys. The help text also provides some
stats, such as number of lights used. For futher stats and performance measures
press <F2>, which shows a profile tree, with counters and timers. All timings 
are in milliseconds.

To change the maximum number of lights supported and other compile time 
options, see Config.h.


Shading Models
----------------
The shading models are implemented in separate files named 
'shaders/shadingModels/ShadingModel_*.glsl'. These are included by the file
'shaders/ShadingModel.glsl', which at compile time or run time selects the 
model(s) needed.

To add a new shading model, implement in a shader file as the existing ones.
Then add an enumeration using a define 'shaders/ShadingModel.glsl', e.g.:
	#define SHADING_MODEL_MEGASHADER 6
and include as illustrated by the existing shading models, e.g.:
	#if UBERSHADER || SHADING_MODEL == SHADING_MODEL_MEGASHADER
		#include "shadingModels/ShadingModel_MegaShader.glsl"
	#endif
The UBERSHADER thing is present to allows the tiled deferred to compile all the
shading models into a single shader and select at run-time. The forward shader
onle ever compiles the used shading model for the current material.
	
The 'chag_shading_model' token from the mtl file, see below, is used to select
the shading model and must match the define, e.g.:
	chag_shading_model SHADING_MODEL_MEGASHADER


OBJ Format Usage/Extension
----------------------
We have added the token 'chag_shading_model <string-id>' to the mtl format that
goes with OBJ files, e.g.:

newmtl sphere1_mat1SG
illum 4
Kd 0.00 1.00 0.00
Ka 0.00 0.00 0.00
d 0.5
Ni 1.00
Ks 0.31 0.31 0.31
Ns 398.00
chag_shading_model SHADING_MODEL_CARPAINT

'd' is used to indicate transparency, or alpha, value. We also parse 
'Tf <r> <g> <b>' and use the average as alpha value.



Programmer Guide
------------------
The most interesting files ought to be:

'tiled_forward_shading_demo.cpp' - contains the main program logic, and is admittedly
a bit of a monster. The rendering is controlled from the function onGlutDisplay,
so why not start there? The enum RenderMethod, is used to select which type of shading
is performed so that can help guide your reading.

'shaders/LightAccel_Tiled.glsl' - in which all the logic and uniforms needed to 
compute tiled shading, in the shaders, is contained. This file is included from
'tiled_forward_fragment.glsl' and 'tiled_deferred_fragment.glsl', and there
used to compute shading. 

'LightGrid.h/cpp' - contains the logic needed to construct the light grid on the
CPU.

'Config.h' - as noted before, this is where some of the core program behaviour
can be configured. For example, maximum number of lights and grid resolution.
Note that these properties may be subject to hardware/API restrictions, read
associated comments carefully.

'CudaRenderer.cu' - is the place where the optimized CUDA version of tiled deferred
shading resides. This code is not necessary to understand the basic tiled shading 
algorithm, but contains several optimizations. Among them are a much faster depth 
buffer min/max reduction and the MSAA optimizations suggested by Lauritzen [4]. 
Note however, that while this is quick at shading, it incurs an extra step to 
copy the MSAA G-Buffers into buffers that can be mapped into CUDA. This is hideously
expensive, both in terms of storage and performance. Thus it is only relevant to 
compare shading performance. An implementation in DX11 can, and should, directly 
access the MSAA G-Buffers.


Buiding the source
--------------------
This should be as easy as opening the solution (tiled_forward_shading_demo.sln 
for Visual Studio 2008, and tiled_forward_shading_demo_2010.sln for Visual 
studio 2010) and pressing whatever button does the build for you. There are 
binaries and libraries for 32-bit and 64-bit targets.

No other OS or environment is directly supported. However, most of the source
is standard C++ and most, if not all, exist with a linux port, so a conversion 
should be relatively straight forward. The libraries that the demo depends on 
must be acquired independently. These are: DevIL, freeglut, glew, and CUDA. 


System Requirements:
-----------------------
Windows XP or above
Graphics Card + Driver supporting OpenGL 3.3


References
------------
[1]	Tiled Shading, Ola Olsson and Ulf Assarsson
	http://www.cse.chalmers.se/~olaolss/jgt2011/

[2] Clustered Deferred and Forward Shading, Ola Olsson, Markus Billeter and Ulf Assarsson
	http://www.cse.chalmers.se/~olaolss/main_frame.php?contents=publication&id=clustered_shading

[3] Tiled and Clustered Forward Shading, Ola Olsson, Markus Billeter and Ulf Assarsson
	http://www.cse.chalmers.se/~olaolss/main_frame.php?contents=publication&id=tiled_clustered_forward_talk

[4]	Deferred Rendering for Current and Future Rendering Pipelines, Andrew Lauritzen
	http://software.intel.com/en-us/articles/deferred-rendering-for-current-and-future-rendering-pipelines/

[5] 

[6]
