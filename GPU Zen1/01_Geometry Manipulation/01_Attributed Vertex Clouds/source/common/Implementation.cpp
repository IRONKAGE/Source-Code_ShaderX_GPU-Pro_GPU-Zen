
#include "Implementation.h"


Implementation::Implementation(const std::string & name)
: m_name(name)
, m_initialized(false)
{
}

Implementation::~Implementation()
{
}

const std::string & Implementation::name() const
{
    return m_name;
}

bool Implementation::initialized() const
{
    return m_initialized;
}

void Implementation::initialize()
{
    if (!initialized())
    {
        onInitialize();

        m_initialized = true;
    }
}

void Implementation::render()
{
    initialize();

    onRender();
}

size_t Implementation::fullByteSize() const
{
    return byteSize() + staticByteSize();
}
