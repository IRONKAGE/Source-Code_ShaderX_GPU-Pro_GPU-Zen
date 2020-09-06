#include "PerfTreeBuilder.h"
#include <algorithm>
#include <utility>


std::vector<PerfTreeNode*> PerfTreeBuilder::build(Profiler &profiler)
{
	Profiler &p = Profiler::instance();
	Profiler::const_iterator it = p.begin();
	std::vector<int> path;

	for (PathItemMap::iterator it = m_pathItemMap.begin(); it != m_pathItemMap.end(); ++it)
	{
		PerfTreeNode *n = (*it).second;

		n->averageSum -= n->averageBuffer[n->currentAverageSlot % PerfTreeNode::s_numAvgSlots];
		// store new value to be part of average.
		n->averageBuffer[n->currentAverageSlot % PerfTreeNode::s_numAvgSlots] = n->time;
		// add new to sum to be averaged
		n->averageSum += n->time;
		n->average = n->averageSum / double(std::min<uint32_t>(n->currentAverageSlot + 1, PerfTreeNode::s_numAvgSlots));
		n->currentAverageSlot++;
		n->count = 0ULL;
		n->time = 0.0;
	}

	return buildItemTree(it, p, path);
}



std::vector<PerfTreeNode*> PerfTreeBuilder::buildItemTree(Profiler::const_iterator &it, Profiler &profiler, std::vector<int> &path)
{
  std::vector<PerfTreeNode*> items;

  double startTime = (*it).m_timeStamp;

  for (; it != profiler.end(); ++it)
  {
    bool stop = false;

    switch((*it).m_id)
    {
    case Profiler::CID_Message:
      break;
    case Profiler::CID_Counter:
      {
        path.push_back((*it).m_tokenId);

				PerfTreeNode *item = m_pathItemMap[path];
				if (!item)
				{
					item = new PerfTreeNode(profiler.getTokenString((*it).m_tokenId));
					m_pathItemMap[path] = item;
				}

        item->count += uint64_t((*it).m_value);
        path.pop_back();

				if (std::find(items.begin(), items.end(), item) == items.end())
        {
					items.push_back(item);
        }
      }
      break;
    case Profiler::CID_BeginBlock:
      {
        PerfTreeNode *item = handleBlock(it, profiler, path);
				if (std::find(items.begin(), items.end(), item) == items.end())
        {
					items.push_back(item);
        }
      }
      break;
    case Profiler::CID_EndBlock:
      return items;
    case Profiler::CID_OneOff:
      break;
    };
  }

  return items;
}




PerfTreeNode *PerfTreeBuilder::handleBlock(Profiler::const_iterator &it, Profiler &profiler, std::vector<int> &path)
{
  double startTime = (*it).m_timeStamp;
  ASSERT((*it).m_id == Profiler::CID_BeginBlock);
	path.push_back((*it).m_tokenId);
  PerfTreeNode *result = m_pathItemMap[path];
  if (!result)
  {
    result = new PerfTreeNode(profiler.getTokenString((*it).m_tokenId));
    m_pathItemMap[path] = result;
  }
  result->count += 1;
  ++it;

	result->children = buildItemTree(it, profiler, path);
  path.pop_back();
  result->time += (*it).m_timeStamp - startTime;
  ASSERT((*it).m_id == Profiler::CID_EndBlock);

  return result;
}
