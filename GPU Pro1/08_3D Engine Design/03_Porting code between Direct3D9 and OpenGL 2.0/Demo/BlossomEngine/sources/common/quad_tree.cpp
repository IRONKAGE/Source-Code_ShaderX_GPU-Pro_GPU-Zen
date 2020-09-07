/* $Id: quad_tree.cpp 247 2009-09-08 08:57:56Z chomzee $ */

#include <stdio.h>

#include "../math/blossom_engine_math.h"
#include "quad_tree.h"

// ----------------------------------------------------------------------------

namespace Blossom
{
	void CQuadTree::initNode(QuadTreeNode *&node, int level, const CVector2 &min, const CVector2 &max)
	{
		if (level > 0)
		{
			node = new QuadTreeNode();

			node->min = min;
			node->max = max;
			node->level = level;
			node->visible = false;

			node->leftUp = NULL;
			node->leftDown = NULL;
			node->rightUp = NULL;
			node->rightDown = NULL;

			initNode(node->leftUp, level - 1, min, 0.5f*(min + max));
			initNode(node->leftDown, level - 1, CVector2(min.x, min.y + 0.5f*(max.y - min.y)), CVector2(min.x + 0.5f*(max.x - min.x), max.y));
			initNode(node->rightUp, level - 1, CVector2(min.x + 0.5f*(max.x - min.x), min.y), CVector2(max.x, min.y + 0.5f*(max.y - min.y)));
			initNode(node->rightDown, level - 1, 0.5f*(min + max), max);
		}
	}



	void CQuadTree::freeNode(QuadTreeNode *&node)
	{
		if (node->leftUp != NULL)
			freeNode(node->leftUp);
		if (node->leftDown != NULL)
			freeNode(node->leftDown);
		if (node->rightUp != NULL)
			freeNode(node->rightUp);
		if (node->rightDown != NULL)
			freeNode(node->rightDown);

		delete node;
		node = NULL;
	}



	void CQuadTree::assignEntityToNode(QuadTreeNode *&node, QuadTreeEntity *entity)
	{
		if ( ( (entity->min.x >= node->min.x) &&
			   (entity->max.x <= node->max.x) &&
 			   (entity->min.y >= node->min.y) &&
			   (entity->max.y <= node->max.y) ) )
		{
			entity->node = node;

			if (node->leftUp != NULL)
				assignEntityToNode(node->leftUp, entity);
			if (node->leftDown != NULL)
				assignEntityToNode(node->leftDown, entity);
			if (node->rightUp != NULL)
				assignEntityToNode(node->rightUp, entity);
			if (node->rightDown != NULL)
				assignEntityToNode(node->rightDown, entity);
		}
	}



	void CQuadTree::resetNodeVisibility(QuadTreeNode *&node)
	{
		node->visible = false;

		if (node->leftUp != NULL)
			resetNodeVisibility(node->leftUp);
		if (node->leftDown != NULL)
			resetNodeVisibility(node->leftDown);
		if (node->rightUp != NULL)
			resetNodeVisibility(node->rightUp);
		if (node->rightDown != NULL)
			resetNodeVisibility(node->rightDown);
	}



	void CQuadTree::updateNodeVisibility(QuadTreeNode *&node, const CPlane planes[], int planesNum)
	{
		CVector3 nodeCorners[8];

		{
			nodeCorners[0] = CVector3(
				node->min.x,
				minHeight,
				node->min.y);
			nodeCorners[1] = CVector3(
				node->min.x,
				maxHeight,
				node->min.y);
		}
		{
			nodeCorners[2] = CVector3(
				node->min.x,
				minHeight,
				node->max.y);
			nodeCorners[3] = CVector3(
				node->min.x,
				maxHeight,
				node->max.y);
		}
		{
			nodeCorners[4] = CVector3(
				node->max.x,
				minHeight,
				node->min.y);
			nodeCorners[5] = CVector3(
				node->max.x,
				maxHeight,
				node->min.y);
		}
		{
			nodeCorners[6] = CVector3(
				node->max.x,
				minHeight,
				node->max.y);
			nodeCorners[7] = CVector3(
				node->max.x,
				maxHeight,
				node->max.y);
		}

		{
			int i;

			for (i = 0; i < planesNum; i++)
			{
				if ( (planes[i].getSignedDistanceFromPoint(nodeCorners[0]) > 0.0f &&
					  planes[i].getSignedDistanceFromPoint(nodeCorners[1]) > 0.0f &&
					  planes[i].getSignedDistanceFromPoint(nodeCorners[2]) > 0.0f &&
					  planes[i].getSignedDistanceFromPoint(nodeCorners[3]) > 0.0f &&
					  planes[i].getSignedDistanceFromPoint(nodeCorners[4]) > 0.0f &&
					  planes[i].getSignedDistanceFromPoint(nodeCorners[5]) > 0.0f &&
					  planes[i].getSignedDistanceFromPoint(nodeCorners[6]) > 0.0f &&
					  planes[i].getSignedDistanceFromPoint(nodeCorners[7]) > 0.0f) )
				{
					break;
				}
			}

			if (i == planesNum)
			{
				node->visible = true;

				if (node->leftUp != NULL)
					updateNodeVisibility(node->leftUp, planes, planesNum);
				if (node->leftDown != NULL)
					updateNodeVisibility(node->leftDown, planes, planesNum);
				if (node->rightUp != NULL)
					updateNodeVisibility(node->rightUp, planes, planesNum);
				if (node->rightDown != NULL)
					updateNodeVisibility(node->rightDown, planes, planesNum);
			}
		}
	}



	void CQuadTree::init(int levels, const CVector2 &min, const CVector2 &max, float minHeight, float maxHeight)
	{
		initNode(root, levels, min, max);

		this->levels = levels;

		this->minHeight = minHeight;
		this->maxHeight = maxHeight;
	}

	
	
	void CQuadTree::free()
	{
		freeNode(root);
	}



	void CQuadTree::assignEntity(QuadTreeEntity *entity)
	{
		assignEntityToNode(root, entity);
	}



	void CQuadTree::resetVisibility()
	{
		resetNodeVisibility(root);
	}



	void CQuadTree::updateVisibility(const CPlane planes[], int planesNum)
	{
		updateNodeVisibility(root, planes, planesNum);
	}
}
