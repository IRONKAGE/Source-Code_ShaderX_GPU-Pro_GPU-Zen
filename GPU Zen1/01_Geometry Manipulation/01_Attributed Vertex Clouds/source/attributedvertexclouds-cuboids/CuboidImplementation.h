
#pragma once

#include "Cuboid.h"
#include "Implementation.h"


class CuboidImplementation : public Implementation
{
public:
    CuboidImplementation(const std::string & name);
    virtual ~CuboidImplementation();

    virtual void setCube(size_t index, const Cuboid & cuboid) = 0;
};
