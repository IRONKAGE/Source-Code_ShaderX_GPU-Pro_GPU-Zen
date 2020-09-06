#ifndef _PerfTreeNode_h_
#define _PerfTreeNode_h_

#include <vector>
#include <map>
#include <string>
#include <utils/IntTypes.h>
#include "profiler/Profiler.h"

struct PerfTreeNode
{
	enum
	{
		s_numAvgSlots = 10,
	};

	PerfTreeNode(const std::string &_label);

	
	std::vector<PerfTreeNode*> children;
	double time;
	std::string label;
	uint64_t count;
	double averageBuffer[s_numAvgSlots];
	double averageSum;
	double average;
	uint32_t currentAverageSlot;
};

#endif // _PerfTreeNode_h_
