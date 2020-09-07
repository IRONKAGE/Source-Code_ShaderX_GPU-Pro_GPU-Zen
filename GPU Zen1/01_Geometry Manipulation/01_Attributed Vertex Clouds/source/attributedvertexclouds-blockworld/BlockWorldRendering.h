
#include <glbinding/gl/types.h>

#include "Rendering.h"


class BlockWorldRendering : public Rendering
{
public:
    BlockWorldRendering();
    virtual ~BlockWorldRendering();

    void increaseBlockThreshold();
    void decreaseBlockThreshold();

protected:
    gl::GLuint m_terrainTexture;

    int m_blockThreshold;

    virtual void onInitialize() override;
    virtual void onDeinitialize() override;
    virtual void onCreateGeometry() override;
    virtual void onPrepareRendering() override;
    virtual void onFinalizeRendering() override;
};
