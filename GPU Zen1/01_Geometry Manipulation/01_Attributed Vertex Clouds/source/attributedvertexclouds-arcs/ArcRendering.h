
#include <glbinding/gl/types.h>

#include "Rendering.h"


class ArcRendering : public Rendering
{
public:
    ArcRendering();
    virtual ~ArcRendering();

protected:
    gl::GLuint m_gradientTexture;

    virtual void onInitialize() override;
    virtual void onDeinitialize() override;
    virtual void onCreateGeometry() override;
    virtual void onPrepareRendering() override;
    virtual void onFinalizeRendering() override;
};
