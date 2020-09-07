
#pragma once

#include "Polygon.h"
#include "Implementation.h"


class PolygonImplementation : public Implementation
{
public:
    PolygonImplementation(const std::string & name);
    virtual ~PolygonImplementation();

    virtual void setPolygon(size_t index, const Polygon & polygon) = 0;
};
