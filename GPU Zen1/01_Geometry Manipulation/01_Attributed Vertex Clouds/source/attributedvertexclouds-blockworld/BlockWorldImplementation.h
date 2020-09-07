
#pragma once


#include "Block.h"
#include "Implementation.h"


class BlockWorldImplementation : public Implementation
{
public:
    BlockWorldImplementation(const std::string & name);
    virtual ~BlockWorldImplementation();

    void setBlockSize(float size);

    virtual void setBlock(size_t index, const Block & block) = 0;

protected:
    float m_blockSize;
};
