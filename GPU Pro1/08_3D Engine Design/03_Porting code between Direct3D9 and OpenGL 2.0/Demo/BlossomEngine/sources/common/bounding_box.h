/* $Id: bounding_box.h 126 2009-08-22 17:08:39Z maxest $ */

#ifndef _BLOSSOM_ENGINE_BOUNDING_BOX_
#define _BLOSSOM_ENGINE_BOUNDING_BOX_

#include "../math/blossom_engine_math.h"

// ----------------------------------------------------------------------------

namespace Blossom
{
	class CBoundingBox
	{
	public:
		CVector3 min, max;

	public:
		void getCorners(CVector3 corners[8]);
		void getCorners(CVector4 corners[8]);

		void updateWithCorners(const CVector3 corners[8]);
		void updateWithCorners(const CVector4 corners[8]);

		void transform(const CMatrix &transform);

		bool collideWithSetOfPoints(const CVector3 points[], int pointsNum);
	};
}

// ----------------------------------------------------------------------------

#endif
