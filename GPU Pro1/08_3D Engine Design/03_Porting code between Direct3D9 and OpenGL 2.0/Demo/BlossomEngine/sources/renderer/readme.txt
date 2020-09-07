$Id: readme.txt 281 2009-09-11 01:56:17Z maxest $

- D3D renderer uses Shader Model 3.0
- OGL renderer uses NV40 profile for NVidia GPUs (GeForce 6k and up) and uses ARB profile for all other GPUs (depending on hardware shaders will or will not work)
- CRenderer: setRenderTarget, setVertexDeclaration, setShader (for both vertex and pixel) must be called at least once per frame to handle lost device correctly (only for Direct3D)
- CRenderer::setVertexDeclaration must be called before CRenderer::setVertexBuffer
- do not use any perspective transformation in meshes' world transform matrices
