#include "light_definitions.h"


StructuredBuffer<SFiniteLightBound> g_data : register( t0 );



#define FLT_EPSILON     1.192092896e-07F        // smallest such that 1.0+FLT_EPSILON != 1.0
#define NR_THREADS			64

// output buffer
RWStructuredBuffer<float3> g_vBoundsBuffer : register( u0 );

#define MAX_PNTS		9		// strictly this should be 10=6+4 but we get more wavefronts and 10 seems to never hit (fingers crossed)
								// However, worst case the plane that would be skipped if such an extreme case ever happened would be backplane
								// clipping gets skipped which doesn't cause any errors.


// LDS (2496 bytes)
groupshared float posX[MAX_PNTS*8*2];
groupshared float posY[MAX_PNTS*8*2];
groupshared float posZ[MAX_PNTS*8*2];
groupshared float posW[MAX_PNTS*8*2];
groupshared unsigned int clipFlags[48];


unsigned int GetClip(const float4 P);
int ClipAgainstPlane(const int iSrcIndex, const int iNrSrcVerts, const int subLigt, const int p);
void CalcBound(out bool2 bIsMinValid, out bool2 bIsMaxValid, out float2 vMin, out float2 vMax, float4x4 InvProjection, float3 pos_view_space, float r);
void GetQuad(out float3 p0, out float3 p1, out float3 p2, out float3 p3, const float3 vBoxX, const float3 vBoxY, const float3 vBoxZ, const float3 vCen, const float2 vScaleXZ, const int sideIndex);


[numthreads(NR_THREADS, 1, 1)]
void main(uint threadID : SV_GroupIndex, uint3 u3GroupID : SV_GroupID)
{
	uint groupID = u3GroupID.x;
	
	//uint vindex = groupID * NR_THREADS + threadID;
	unsigned int g = groupID;
	unsigned int t = threadID;

	const int subLigt = (int) (t/8);
	const int lgtIndex = subLigt+(int) g*8;
	const int sideIndex = (int) (t%8);
	
	SFiniteLightBound lgtDat = g_data[lgtIndex];
	
	const float3 vBoxX = lgtDat.vBoxAxisX.xyz;
	const float3 vBoxY = lgtDat.vBoxAxisY.xyz;
	const float3 vBoxZ = lgtDat.vBoxAxisZ.xyz;
	const float3 vCen = lgtDat.vCen.xyz;
	const float fRadius = lgtDat.fRadius;
	const float2 vScaleXZ = lgtDat.vScaleXZ;

	{
		if(sideIndex<6 && lgtIndex<g_iNrVisibLights)		// mask 2 out of 8 threads
		{
			float3 q0, q1, q2, q3;
			GetQuad(q0, q1, q2, q3, vBoxX, vBoxY, vBoxZ, vCen, vScaleXZ, sideIndex);


			const float4 vP0 = mul(float4(q0, 1), g_mProjection);
			const float4 vP1 = mul(float4(q1, 1), g_mProjection);
			const float4 vP2 = mul(float4(q2, 1), g_mProjection);
			const float4 vP3 = mul(float4(q3, 1), g_mProjection);

			// test vertices of one quad (of the convex hull) for intersection
			const unsigned int uFlag0 = GetClip(vP0);
			const unsigned int uFlag1 = GetClip(vP1);
			const unsigned int uFlag2 = GetClip(vP2);
			const unsigned int uFlag3 = GetClip(vP3);

			const float4 vPnts[] = {vP0, vP1, vP2, vP3};
				
			// screen-space AABB of one quad (assuming no intersection)
			float3 vMin, vMax;
			for(int k=0; k<4; k++)
			{
				float fW = vPnts[k].w;
				float fS = fW<0 ? -1 : 1;
				float fWabs = fW<0 ? (-fW) : fW;
				fW = fS * (fWabs<FLT_EPSILON ? FLT_EPSILON : fWabs);
				float3 vP = float3(vPnts[k].x/fW, vPnts[k].y/fW, vPnts[k].z/fW);
				if(k==0) { vMin=vP; vMax=vP; }
				
				vMax = max(vMax, vP); vMin = min(vMin, vP);
			}

			clipFlags[subLigt*6+sideIndex] = (uFlag0<<0) | (uFlag1<<6) | (uFlag2<<12) | (uFlag3<<18);

			// store in clip buffer (only use these vMin and vMax if light is 100% visible in which case clipping isn't needed)
			posX[subLigt*MAX_PNTS*2 + sideIndex] = vMin.x;
			posY[subLigt*MAX_PNTS*2 + sideIndex] = vMin.y;
			posZ[subLigt*MAX_PNTS*2 + sideIndex] = vMin.z;

			posX[subLigt*MAX_PNTS*2 + sideIndex + 6] = vMax.x;
			posY[subLigt*MAX_PNTS*2 + sideIndex + 6] = vMax.y;
			posZ[subLigt*MAX_PNTS*2 + sideIndex + 6] = vMax.z;
		}
	}

	// if not XBONE and not PLAYSTATION4 we need a memorybarrier here
	// since we can't rely on the gpu cores being 64 wide.
	// We need a pound define around this.
	GroupMemoryBarrierWithGroupSync();


	{
		if(sideIndex==0 && lgtIndex<g_iNrVisibLights)
		{
			// quick acceptance or rejection
			unsigned int uCollectiveAnd = (unsigned int) -1;
			unsigned int uCollectiveOr = 0;
			for(int f=0; f<6; f++)
			{
				unsigned int uFlagAnd = clipFlags[subLigt*6+f]&0x3f;
				unsigned int uFlagOr = uFlagAnd;
				for(int i=1; i<4; i++)
				{
					unsigned int uClipBits = (clipFlags[subLigt*6+f]>>(i*6))&0x3f;
					uFlagAnd &= uClipBits;
					uFlagOr |= uClipBits;
				}

				uCollectiveAnd &= uFlagAnd;
				uCollectiveOr |= uFlagOr;
			}

			bool bSetBoundYet = false;
			float3 vMin, vMax;
			if(uCollectiveAnd!=0 || uCollectiveOr==0)		// all invisible or all visible (early out)
			{
				if(uCollectiveOr==0)	// all visible
				{
					for(int f=0; f<6; f++)
					{
						const int sideIndex = f;

						float3 vFaceMi = float3(posX[subLigt*MAX_PNTS*2 + sideIndex + 0], posY[subLigt*MAX_PNTS*2 + sideIndex + 0], posZ[subLigt*MAX_PNTS*2 + sideIndex + 0]);
						float3 vFaceMa = float3(posX[subLigt*MAX_PNTS*2 + sideIndex + 6], posY[subLigt*MAX_PNTS*2 + sideIndex + 6], posZ[subLigt*MAX_PNTS*2 + sideIndex + 6]);
						
						for(int k=0; k<2; k++)
						{
							float3 vP = k==0 ? vFaceMi : vFaceMa;
							if(f==0 && k==0) { vMin=vP; vMax=vP; }
							
							vMax = max(vMax, vP); vMin = min(vMin, vP);
						}
					}
					bSetBoundYet=true;
				}
			}
			else		// :( need true clipping
			{
				
				for(int f=0; f<6; f++)
				{
					float3 q0, q1, q2, q3;
					GetQuad(q0, q1, q2, q3, vBoxX, vBoxY, vBoxZ, vCen, vScaleXZ, f);
			
					// 4 vertices to a quad of the convex hull in post projection space
					const float4 vP0 = mul(float4(q0, 1), g_mProjection);
					const float4 vP1 = mul(float4(q1, 1), g_mProjection);
					const float4 vP2 = mul(float4(q2, 1), g_mProjection);
					const float4 vP3 = mul(float4(q3, 1), g_mProjection);

					
					int iSrcIndex = 0;

					int offs = iSrcIndex*MAX_PNTS+subLigt*MAX_PNTS*2;

					// fill up source clip buffer with the quad
					posX[offs+0]=vP0.x; posX[offs+1]=vP1.x; posX[offs+2]=vP2.x; posX[offs+3]=vP3.x;
					posY[offs+0]=vP0.y; posY[offs+1]=vP1.y; posY[offs+2]=vP2.y; posY[offs+3]=vP3.y;
					posZ[offs+0]=vP0.z; posZ[offs+1]=vP1.z; posZ[offs+2]=vP2.z; posZ[offs+3]=vP3.z;
					posW[offs+0]=vP0.w; posW[offs+1]=vP1.w; posW[offs+2]=vP2.w; posW[offs+3]=vP3.w;

					int iNrSrcVerts = 4;

					// do true clipping
					for(int p=0; p<6; p++)
					{
						const int nrVertsDst = ClipAgainstPlane(iSrcIndex, iNrSrcVerts, subLigt, p);

						iSrcIndex = 1-iSrcIndex;
						iNrSrcVerts = nrVertsDst;

						if(iNrSrcVerts<3 || iNrSrcVerts>=MAX_PNTS) break;
					}

					// final clipped convex primitive is in src buffer
					if(iNrSrcVerts>2)
					{
						int offs_src = iSrcIndex*MAX_PNTS+subLigt*MAX_PNTS*2;
						for(int k=0; k<iNrSrcVerts; k++)
						{
							float4 vCur = float4(posX[offs_src+k], posY[offs_src+k], posZ[offs_src+k], posW[offs_src+k]);
							
							// project and apply toward AABB
							float3 vP = float3(vCur.x/vCur.w, vCur.y/vCur.w, vCur.z/vCur.w);
							if(!bSetBoundYet) { vMin=vP; vMax=vP; bSetBoundYet=true; }
							
							vMax = max(vMax, vP); vMin = min(vMin, vP);
						}
					}
						
				}

				////////////////////// look for camera frustum verts that need to be included. That is frustum vertices inside the convex hull for the light

				for(int i=0; i<8; i++)	// establish 8 camera frustum vertices
				{
					float3 vVertPSpace = float3((i&1)!=0 ? 1 : (-1), (i&2)!=0 ? 1 : (-1), (i&4)!=0 ? 1 : 0);
				
					float4 v4ViewSpace = mul(float4(vVertPSpace,1), g_mInvProjection);
					float3 vViewSpace = float3(v4ViewSpace.x/v4ViewSpace.w, v4ViewSpace.y/v4ViewSpace.w, v4ViewSpace.z/v4ViewSpace.w);

					posX[subLigt*MAX_PNTS*2 + i] = vViewSpace.x;
					posY[subLigt*MAX_PNTS*2 + i] = vViewSpace.y;
					posZ[subLigt*MAX_PNTS*2 + i] = vViewSpace.z;
				}

				// determine which camera frustum vertices are inside the convex hull
				uint uVisibFl = 0xff;
				for(int f=0; f<6; f++)
				{
					float3 vP0, vP1, vP2, vP3;
					GetQuad(vP0, vP1, vP2, vP3, vBoxX, vBoxY, vBoxZ, vCen, vScaleXZ, f);

					// one edge might be zero length so we do all 4
					float3 vN = cross(vP1-vP0, vP3-vP0) + cross(vP2-vP1, vP0-vP1) + cross(vP3-vP2, vP1-vP2) + cross(vP0-vP3, vP2-vP3);
					float fLen = length(vN);
					if(fLen>1) vN = normalize(vN);		// this won't necessarily be a non zero vector (spot lights have all 4 top points as the same)

					for(int i=0; i<8; i++)
					{
						float3 vViewSpace = float3(posX[subLigt*MAX_PNTS*2 + i], posY[subLigt*MAX_PNTS*2 + i], posZ[subLigt*MAX_PNTS*2 + i]);
#ifdef LEFT_HAND_COORDINATES
						uVisibFl &= ( dot(vViewSpace-vP0, vN)<0 ? 0xff : (~(1<<i)) );
#else
						uVisibFl &= ( dot(vViewSpace-vP0, vN)>0 ? 0xff : (~(1<<i)) );
#endif
					}
				}

				// apply camera frustum vertices inside the convex hull to the AABB
				for(int i=0; i<8; i++)
				{
					if((uVisibFl&(1<<i))!=0)
					{
						float3 vP = float3((i&1)!=0 ? 1 : (-1), (i&2)!=0 ? 1 : (-1), (i&4)!=0 ? 1 : 0);

						if(!bSetBoundYet) { vMin=vP; vMax=vP; bSetBoundYet=true; }
							
						vMax = max(vMax, vP); vMin = min(vMin, vP);
					}
				}
			}


			


			// determine AABB bound in [-1;1]x[-1;1] screen space using bounding sphere.
			// Use the result to make our already established AABB from the convex hull
			// potentially tighter.
			if(!bSetBoundYet)
			{
				// set the AABB off-screen
				vMin = float3(-3,-3,-3);
				vMax = float3(-2,-2,-2);
			}
			else
			{
				//if((vCen.z+fRadius)<0.0)
				if( length(vCen)>fRadius)
				{
					float2 vMi, vMa;
					bool2 bMi, bMa;
					CalcBound(bMi, bMa, vMi, vMa, g_mInvProjection, vCen, fRadius);
		
					vMin.xy = bMi ? max(vMin.xy, vMi) : vMin.xy;
					vMax.xy = bMa ? min(vMax.xy, vMa) : vMax.xy;
				}

#ifdef LEFT_HAND_COORDINATES
				if((vCen.z-fRadius)>0.0)
				{
					float4 vPosF = mul(float4(0,0,vCen.z-fRadius,1), g_mProjection);
					vMin.z = max(vMin.z, vPosF.z/vPosF.w);
				}
				if((vCen.z+fRadius)>0.0)
				{
					float4 vPosB = mul(float4(0,0,vCen.z+fRadius,1), g_mProjection);
					vMax.z = min(vMax.z, vPosB.z/vPosB.w);
				}
#else
				if((vCen.z+fRadius)<0.0)
				{
					float4 vPosF = mul(float4(0,0,vCen.z+fRadius,1), g_mProjection);
					vMin.z = max(vMin.z, vPosF.z/vPosF.w);
				}
				if((vCen.z-fRadius)<0.0)
				{
					float4 vPosB = mul(float4(0,0,vCen.z-fRadius,1), g_mProjection);
					vMax.z = min(vMax.z, vPosB.z/vPosB.w);
				}
#endif
				else
				{
					vMin = float3(-3,-3,-3);
					vMax = float3(-2,-2,-2);
				}
			}


			// we should consider doing a look-up here into a max depth mip chain
			// to see if the light is occluded: vMin.z*VIEWPORT_SCALE_Z > MipTexelMaxDepth
			g_vBoundsBuffer[lgtIndex+0] = float3(0.5*vMin.x+0.5, -0.5*vMax.y+0.5, vMin.z*VIEWPORT_SCALE_Z);
			g_vBoundsBuffer[lgtIndex+g_iNrVisibLights] = float3(0.5*vMax.x+0.5, -0.5*vMin.y+0.5, vMax.z*VIEWPORT_SCALE_Z);
		}
	}
}


float4 GenNewVert(const float4 vVisib, const float4 vInvisib, const int p);

int ClipAgainstPlane(const int iSrcIndex, const int iNrSrcVerts, const int subLigt, const int p)
{
	int offs_src = iSrcIndex*MAX_PNTS+subLigt*MAX_PNTS*2;
	int offs_dst = (1-iSrcIndex)*MAX_PNTS+subLigt*MAX_PNTS*2;

	float4 vPrev = float4(posX[offs_src+(iNrSrcVerts-1)], posY[offs_src+(iNrSrcVerts-1)], posZ[offs_src+(iNrSrcVerts-1)], posW[offs_src+(iNrSrcVerts-1)]);

	int nrVertsDst = 0;

	unsigned int uMask = (1<<p);
	bool bIsPrevVisib = (GetClip(vPrev)&uMask)==0;
	for(int i=0; i<iNrSrcVerts; i++)
	{
		float4 vCur = float4(posX[offs_src+i], posY[offs_src+i], posZ[offs_src+i], posW[offs_src+i]);
		bool bIsCurVisib = (GetClip(vCur)&uMask)==0;
		if( (bIsCurVisib && !bIsPrevVisib) || (!bIsCurVisib && bIsPrevVisib) )
		{
			//assert(nrVertsDst<MAX_PNTS);
			if(nrVertsDst<MAX_PNTS)
			{
				// generate new vertex
				float4 vNew = GenNewVert(bIsCurVisib ? vCur : vPrev, bIsCurVisib ? vPrev : vCur, p);
				posX[offs_dst+nrVertsDst]=vNew.x; posY[offs_dst+nrVertsDst]=vNew.y; posZ[offs_dst+nrVertsDst]=vNew.z; posW[offs_dst+nrVertsDst]=vNew.w;
				++nrVertsDst;
			}
		}
							
		if(bIsCurVisib)
		{
			//assert(nrVertsDst<MAX_PNTS);
			if(nrVertsDst<MAX_PNTS)
			{
				posX[offs_dst+nrVertsDst]=vCur.x; posY[offs_dst+nrVertsDst]=vCur.y; posZ[offs_dst+nrVertsDst]=vCur.z; posW[offs_dst+nrVertsDst]=vCur.w;
				++nrVertsDst;
			}
		}

		vPrev = vCur;
		bIsPrevVisib = bIsCurVisib;
	}

	return nrVertsDst;
}



unsigned int GetClip(const float4 P)
{
	//-P.w <= P.x <= P.w
	return ((P.x<-P.w)?1:0) | ((P.x>P.w)?2:0) | ((P.y<-P.w)?4:0) | ((P.y>P.w)?8:0) | ((P.z<0)?16:0) | ((P.z>P.w)?32:0);
}

float4 GenNewVert(const float4 vVisib, const float4 vInvisib, const int p)
{
	const float fS = p==4 ? 0 : ((p&1)==0 ? -1 : 1);
	const int index = p/2;
	float x1 = index==0 ? vVisib.x : (index==1 ? vVisib.y : vVisib.z);
	float x0 = index==0 ? vInvisib.x : (index==1 ? vInvisib.y : vInvisib.z);
	
	//fS*((vVisib.w-vInvisib.w)*t + vInvisib.w) = (x1-x0)*t + x0;

	const float fT = (fS*vInvisib.w-x0)/((x1-x0) - fS*(vVisib.w-vInvisib.w));
	float4 vNew = vVisib*fT + vInvisib*(1-fT);

	// just to be really anal we make sure the clipped against coordinate is precise
	if(index==0) vNew.x = fS*vNew.w;
	else if(index==1) vNew.y = fS*vNew.w;
	else vNew.z = fS*vNew.w;

	return vNew;
}

void GetQuad(out float3 p0, out float3 p1, out float3 p2, out float3 p3, const float3 vBoxX, const float3 vBoxY, const float3 vBoxZ, const float3 vCen, const float2 vScaleXZ, const int sideIndex)
{
	const int iAbsSide = (sideIndex == 0 || sideIndex == 1) ? 0 : ((sideIndex == 2 || sideIndex == 3) ? 1 : 2);
	const float fS = (sideIndex & 1) != 0 ? 1 : (-1);

	float3 vA = fS*(iAbsSide == 0 ? vBoxX : (iAbsSide == 1 ? vBoxZ : vBoxY));
	float3 vB = fS*(iAbsSide == 0 ? vBoxZ : (iAbsSide == 1 ? (-vBoxX) : vBoxZ));
	float3 vC = iAbsSide == 0 ? vBoxY : (iAbsSide == 1 ? vBoxY : (-vBoxX));

	bool bIsTopQuad = iAbsSide == 2 && (sideIndex & 1) != 0;		// in this case all 4 verts get scaled.
	bool bIsSideQuad = (iAbsSide == 0 || iAbsSide == 1);		// if side quad only two verts get scaled (impacts q1 and q2)

	if (bIsTopQuad) { vB *= vScaleXZ.y; vC *= vScaleXZ.x; }

	float3 vA2 = vA;
	float3 vB2 = vB;

	if (bIsSideQuad) { vA2 *= (iAbsSide == 0 ? vScaleXZ.x : vScaleXZ.y); vB2 *= (iAbsSide == 0 ? vScaleXZ.y : vScaleXZ.x); }

	p0 = vCen + vA + vB - vC;		// vCen + vA is center of face when vScaleXZ is 1.0
	p1 = vCen + vA2 + vB2 + vC;
	p2 = vCen + vA2 - vB2 + vC;
	p3 = vCen + vA - vB - vC;
}



float4 TransformPlaneToPostSpace(float4x4 InvProjection, float4 plane)
{
	return mul(InvProjection, plane);
}

float4 EvalPlanePair(float2 posXY_in, float r)
{
	// rotate by 90 degrees to avoid potential division by zero
	bool bMustFlip = abs(posXY_in.y)<abs(posXY_in.x);
	float2 posXY = bMustFlip ? float2(-posXY_in.y, posXY_in.x) : posXY_in;

	float fLenSQ = dot(posXY, posXY);
	float D = posXY.y * sqrt(fLenSQ - r*r);

	float4 res;
	res.x = (-r*posXY.x - D) / fLenSQ;
	res.z = (-r*posXY.x + D) / fLenSQ;
	res.y = (-r-res.x*posXY.x) / posXY.y;
	res.w = (-r-res.z*posXY.x) / posXY.y;

	// rotate back by 90 degrees
	res = bMustFlip ? Vec4(res.y, -res.x, res.w, -res.z) : res;

	return res;
}

void CalcBound(out bool2 bIsMinValid, out bool2 bIsMaxValid, out float2 vMin, out float2 vMax, float4x4 InvProjection, float3 pos_view_space, float r)
{
	float4 planeX = EvalPlanePair(float2(pos_view_space.x, pos_view_space.z), r);
	float4 planeY = EvalPlanePair(float2(pos_view_space.y, pos_view_space.z), r);


#ifdef LEFT_HAND_COORDINATES
	planeX = planeX.zwxy;		// need to swap left/right and top/bottom planes when using left hand system
	planeY = planeY.zwxy;
#endif

	bIsMinValid = bool2(planeX.z<0, planeY.z<0);
	bIsMaxValid = bool2((-planeX.x)<0, (-planeY.x)<0);

	// hopefully the compiler takes zeros into account
	// should be the case since the transformation in TransformPlaneToPostSpace()
	// is done using multiply-adds and not dot product instructions.
	float4 planeX0 = TransformPlaneToPostSpace(InvProjection, float4(planeX.x, 0, planeX.y, 0));
	float4 planeX1 = TransformPlaneToPostSpace(InvProjection, float4(planeX.z, 0, planeX.w, 0));
	float4 planeY0 = TransformPlaneToPostSpace(InvProjection, float4(0, planeY.x, planeY.y, 0));
	float4 planeY1 = TransformPlaneToPostSpace(InvProjection, float4(0, planeY.z, planeY.w, 0));

	
	// convert planes to the forms (1,0,0,D) and (0,1,0,D)
	// 2D bound is given by -D components
	float2 A = -float2(planeX0.w / planeX0.x, planeY0.w / planeY0.y);
	float2 B = -float2(planeX1.w / planeX1.x, planeY1.w / planeY1.y);

	// Bound is complete
	vMin = B;
	vMax = A;
}
