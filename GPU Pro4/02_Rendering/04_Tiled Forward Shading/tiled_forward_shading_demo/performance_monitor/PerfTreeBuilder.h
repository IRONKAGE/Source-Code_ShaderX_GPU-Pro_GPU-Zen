#ifndef _PerfTreeBuilder_h_
#define _PerfTreeBuilder_h_

#include <vector>
#include <map>
#include <string>
#include <utils/IntTypes.h>
#include "profiler/Profiler.h"

#include "PerfTreeNode.h"


class PerfTreeBuilder
{
public:

	std::vector<PerfTreeNode*> build(Profiler &profiler);

	std::vector<PerfTreeNode*> buildItemTree(Profiler::const_iterator &it, Profiler &profiler, std::vector<int> &path);

protected:
	PerfTreeNode *handleBlock(Profiler::const_iterator &it, Profiler &profiler, std::vector<int> &path);

	typedef std::map<std::vector<int>, PerfTreeNode*> PathItemMap;
	PathItemMap m_pathItemMap;
};

#endif // _PerfTreeBuilder_h_
