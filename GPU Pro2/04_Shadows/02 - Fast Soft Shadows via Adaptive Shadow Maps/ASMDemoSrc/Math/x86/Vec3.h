/* MSVC SSE math library implementation.

   Pavlo Turchyn <pavlo@nichego-novogo.net> Aug 2010 */

#ifndef __VEC3
#define __VEC3

static const IntegerMask c_WMask = {0xffffffff, 0xffffffff, 0xffffffff, 0};
static const __m128 c_WOne = {0, 0, 0, 1};

class Mat4x4;

class Vec3 : public Vec4
{
public:
  finline Vec3()                                  { }
  finline Vec3(__m128 a)                          { r = _mm_and_ps(a, c_WMask.f); }
  finline Vec3(const Vec4& a)                     { r = _mm_and_ps(a.r, c_WMask.f); }
  finline Vec3(float a)                           { r = _mm_and_ps(_mm_set1_ps(a), c_WMask.f); }
  finline Vec3(float a, float b, float c)         { x = a; y = b; z = c; w = 0; }
  finline Vec3(const float* a)                    { x = a[0]; y = a[1]; z = a[2]; w = 0; }

  finline const Vec3& operator = (const Vec3& a)  { r = a.r; return *this; }

  static finline const Vec4 Point(const Vec3& a)  { return _mm_or_ps(a.r, c_WOne); }
  static finline const Vec4 Vector(const Vec3& a) { return a; }

  template<int x, int y, int z> static finline const Vec3 Constant() { static const __m128 r = {(float)x, (float)y, (float)z, 0}; Vec3 v; v.r = r; return v; }

  static finline const Vec3 Cross(const Vec3& a, const Vec3& b)
  {
    __m128 t0 = _mm_mul_ps(SWZ_YZXW(a.r), SWZ_ZXYW(b.r));
    __m128 t1 = _mm_mul_ps(SWZ_ZXYW(a.r), SWZ_YZXW(b.r));
    return _mm_sub_ps(t0, t1);
  }

  static finline const Vec3 Project(const Vec3&, const Mat4x4&);
};

#endif //#ifndef __VEC3
