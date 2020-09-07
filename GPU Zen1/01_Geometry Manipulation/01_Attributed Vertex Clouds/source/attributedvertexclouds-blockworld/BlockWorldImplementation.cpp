
#include "BlockWorldImplementation.h"


BlockWorldImplementation::BlockWorldImplementation(const std::string & name)
: Implementation(name)
, m_blockSize(1.0f)
{
}

BlockWorldImplementation::~BlockWorldImplementation()
{
}

void BlockWorldImplementation::setBlockSize(float size)
{
    m_blockSize = size;
}
