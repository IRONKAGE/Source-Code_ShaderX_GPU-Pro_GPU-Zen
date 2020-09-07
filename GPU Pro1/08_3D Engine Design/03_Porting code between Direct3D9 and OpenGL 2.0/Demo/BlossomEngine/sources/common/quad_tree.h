/* $Id: quad_tree.h 126 2009-08-22 17:08:39Z maxest $ */

#ifndef _BLOSSOM_ENGINE_QUAD_TREE_
#define _BLOSSOM_ENGINE_QUAD_TREE_

// ----------------------------------------------------------------------------

namespace Blossom
{
	class CVideoManager;

	// ----------------------------------------------------------------------------

	struct QuadTreeNode
	{
		CVector2 min, max;
		int level;
		bool visible;
		QuadTreeNode *leftUp, *leftDown, *rightUp, *rightDown;
	};

	struct QuadTreeEntity
	{
		CVector2 min, max;
		const QuadTreeNode *node;
	};

	// ----------------------------------------------------------------------------

	class CQuadTree
	{
		friend class CVideoManager;

	private:
		QuadTreeNode *root;

		int levels;
		float minHeight, maxHeight;

	private:
		void initNode(QuadTreeNode *&node, int level, const CVector2 &min, const CVector2 &max);
		void freeNode(QuadTreeNode *&node);

		void assignEntityToNode(QuadTreeNode *&node, QuadTreeEntity *entity);

		void resetNodeVisibility(QuadTreeNode *&node);
		void updateNodeVisibility(QuadTreeNode *&node, const CPlane planes[], int planesNum);

	public:
		void init(int levels, const CVector2 &min, const CVector2 &max, float minHeight, float maxHeight);
		void free();

		void assignEntity(QuadTreeEntity *entity);

		void resetVisibility();
		void updateVisibility(const CPlane planes[], int planesNum);
	};
}

// ----------------------------------------------------------------------------

#endif
