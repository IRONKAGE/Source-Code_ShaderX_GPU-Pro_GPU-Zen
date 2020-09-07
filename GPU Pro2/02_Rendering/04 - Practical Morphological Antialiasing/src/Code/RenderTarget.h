/*
 * Copyright (C) 2010 Jorge Jimenez (jim@unizar.es)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must display the name 'Jorge Jimenez' as
 *    'Real-Time Rendering R&D' in the credits of the application, if such
 *    credits exist. The author of this work must be notified via email
 *    (jim@unizar.es) in this case of redistribution.
 *
 * 3. Neither the name of copyright holders nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS
 * IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDERS OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef RENDERTARGET_H
#define RENDERTARGET_H

class RenderTarget {
    public:
        static const DXGI_SAMPLE_DESC NO_MSAA;

        RenderTarget(ID3D10Device *device, int width, int height,
            DXGI_FORMAT format,
            const DXGI_SAMPLE_DESC &sampleDesc=NO_MSAA);
        ~RenderTarget();

        operator ID3D10Texture2D * () const { return texture2D; }
        operator ID3D10RenderTargetView * () const { return renderTargetView; }
        operator ID3D10RenderTargetView *const * () const { return &renderTargetView; }
        operator ID3D10ShaderResourceView * () const { return shaderResourceView; }

        int getWidth() const { return width; }
        int getHeight() const { return height; }
        
        void setViewport(float minDepth=0.0f, float maxDepth=1.0f) const;

    private:
        ID3D10Device *device;
        int width, height;
        ID3D10Texture2D *texture2D;
        ID3D10RenderTargetView *renderTargetView;
        ID3D10ShaderResourceView *shaderResourceView;
};

class DepthStencil {
    public:
        static const DXGI_SAMPLE_DESC NO_MSAA;

        DepthStencil(ID3D10Device *device, int width, int height,
            DXGI_FORMAT texture2DFormat = DXGI_FORMAT_R32_TYPELESS, 
            DXGI_FORMAT depthStencilViewFormat = DXGI_FORMAT_D32_FLOAT, 
            DXGI_FORMAT shaderResourceViewFormat = DXGI_FORMAT_R32_FLOAT,
            const DXGI_SAMPLE_DESC &sampleDesc=NO_MSAA);
        ~DepthStencil();

        operator ID3D10Texture2D * const () { return texture2D; }
        operator ID3D10DepthStencilView * const () { return depthStencilView; }
        operator ID3D10ShaderResourceView * const () { return shaderResourceView; }

        int getWidth() const { return width; }
        int getHeight() const { return height; }

        void setViewport(float minDepth=0.0f, float maxDepth=1.0f) const;

    private:
        ID3D10Device *device;
        int width, height;
        ID3D10Texture2D *texture2D;
        ID3D10DepthStencilView *depthStencilView;
        ID3D10ShaderResourceView *shaderResourceView;
};

class Quad {
    public:
        Quad(ID3D10Device *device, const D3D10_PASS_DESC &desc);
        ~Quad();
        void setInputLayout() { device->IASetInputLayout(vertexLayout); }
        void draw();

    private:
        ID3D10Device *device;
        ID3D10Buffer *buffer;
        ID3D10InputLayout *vertexLayout;
};

class SaveViewportScope {
    public: 
        SaveViewportScope(ID3D10Device *device);
        ~SaveViewportScope();

    private:
        ID3D10Device *device;
        D3D10_VIEWPORT viewport;
};

class Utils {
    public:
        static D3D10_VIEWPORT viewportFromView(ID3D10View *view);
        static D3D10_VIEWPORT viewportFromTexture2D(ID3D10Texture2D *texture2D);
};

#endif

