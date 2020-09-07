
#include <glbinding/gl/types.h>

#include "Rendering.h"


class CuboidRendering : public Rendering
{
public:
    CuboidRendering();
    virtual ~CuboidRendering();

protected:
    gl::GLuint m_gradientTexture;

    virtual void onInitialize() override;
    virtual void onDeinitialize() override;
    virtual void onCreateGeometry() override;
    virtual void onPrepareRendering() override;
    virtual void onFinalizeRendering() override;
};
