
#pragma once

#include "Arc.h"
#include "Implementation.h"


class ArcImplementation : public Implementation
{
public:
    ArcImplementation(const std::string & name);
    virtual ~ArcImplementation();

    virtual void setArc(size_t index, const Arc & arc) = 0;
};
