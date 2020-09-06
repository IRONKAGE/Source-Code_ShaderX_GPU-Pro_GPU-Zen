//
// by Jan Eric Kyprianidis <www.kyprianidis.com>
// Copyright (C) 2010-2012 Computer Graphics Systems Group at the
// Hasso-Plattner-Institut, Potsdam, Germany <www.hpi3d.de>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//

Requirements:

    * CUDA 4: http://developer.nvidia.com/cuda/cuda-downloads
    * CMake: http://www.cmake.org
    * Qt: http://qt.nokia.com
    * Libav: http://www.libav.org [optional]

Building:

    Windows / Visual Studio:
        1) mkdir build
        2) cd build
        3) cmake ..
        4) devenv /build Release cefabs.sln

    Mac OS X / Linux:
        1) mkdir build
        2) cd build
        3) cmake ..
        4) make


 Related Publications:

    * Kyprianidis, J. E., & Kang, H. (2011). Image and Video
      Abstraction by Coherence-Enhancing Filtering. Computer Graphics
      Forum, 30(2), 593-602.

    * Kyprianidis, J. E., & Kang, H. (2013). Coherence-Enhancing
      Filtering on the GPU. GPU Pro 4: Advanced Rendering Techniques.
