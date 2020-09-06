#include "PerfTreeNode.h"
#include <algorithm>
#include <memory.h>

PerfTreeNode::PerfTreeNode(const std::string &_label) : 
	label(_label), 
	time(0.0), 
	count(0ULL),
	averageSum(0.0),
	average(0.0),
	currentAverageSlot(0)
{ 
	memset(averageBuffer, 0, sizeof(averageBuffer));
}
