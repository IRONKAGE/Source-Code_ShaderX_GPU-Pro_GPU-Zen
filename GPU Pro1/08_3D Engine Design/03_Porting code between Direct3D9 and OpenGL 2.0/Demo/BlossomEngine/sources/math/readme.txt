$Id: readme.txt 281 2009-09-11 01:56:17Z maxest $

- math module projection matrices assums the clip-space is in [0, 1] range what is valid for D3D, but not for OGL. In fact, after using these functions, z near camer'a plane is a bit shifted toward the viewer in OGL
