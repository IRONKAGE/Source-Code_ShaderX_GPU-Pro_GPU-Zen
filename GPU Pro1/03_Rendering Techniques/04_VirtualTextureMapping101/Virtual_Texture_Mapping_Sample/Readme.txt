Virtual texture mapping demo
----------------------------
Matthäus G. Chajdas <shaderx8@anteru.net>

Introduction
~~~~~~~~~~~~

This is the demo application for the _Virtual texture mapping 101_ chapter.
The code here was used for all illustrations. The main application is in the
+App+ folder. The other folders contain source data, external dependencies
and a small tool to generate input data.

Compiling
~~~~~~~~~

Compiling requires Visual Studio 2008 and a recent DirectX SDK (tested with
March 2009 and August 2009). In order to compile correctly, the Boost headers
must be in the include search path, so they can be found by includes of the
form +#include <boost/foo.hpp>+.

The application is written using DX10, and thus requires Windows Vista. Porting
to Windows XP should be relatively easy as the shader code does not use any
DX10 specific features. The application uses the new Vista thread pool for
all asynchronous operations which have to be mapped to the XP thread pool
calls.

Primary development was done on Vista x64, but the x86 version should run
equally well. The solution contains both configurations.

Running
~~~~~~~

The application contains a few hard-coded paths, in particular, it looks for
a folder named +Source-Data/Tiles/JPEG+ and +Shaders+. The specific locations
that are searched are +../Source-Data/Tiles/JPEG+ and +./Shaders+. The source
drop has the right folder structure. By default, the working directory is set
to the directory containing the project so no modification is required. If the
binary is to be run manually, the +App+ folder has to be selected as the
working directory (required for example for debugging using PIX.)

The pages are named _tile-<level>-<number>_, with _level_ ranging from 0..6,
and number depending on the level. For each level, the tiles are numbered
starting at the top left (0, 0) and counting row-wise (that is, the second
entry is the one right of the first one, and the last one is in the bottom right.)

If everything is set up correctly, opening the solution and selecting one of
the release builds should result in a correctly working binary, which can be
started directly out of Visual Studio using "Run without debugging".

Code
~~~~

The main starting point is in +app.cpp+, which contains the main render loop.
The classes starting with +Render+ are helper classes to simplify the rendering,
and contain little VTM-specific code. Some keys are bound, see +main.cpp+,
+OnKeyboard+ for details.

The DXTC compression is using code from J.M.P. van Waveren, and can be found
in +dxtc.cpp+.

Misc
~~~~

A small C# based Mip-Map generator can be found in +MipMapGenerator+. It expects
a 8192² texture as input, and generates the individual pages with the correct
borders as expected by the application.

[NOTE]
==============================================================================
It fully loads the image for generating mip-maps, and thus requires a few
hundred MiB to run correctly.
==============================================================================