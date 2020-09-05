#ifndef MATRIX4_H
#define MATRIX4_H

// Matrix4
//
// 4x4 matrix in column-major order.
class Matrix4
{
public:
  Matrix4()
  {
    LoadIdentity();
  }

  Matrix4(float entry0, float entry1, float entry2, float entry3,
          float entry4, float entry5, float entry6, float entry7,
          float entry8, float entry9, float entry10, float entry11,
          float entry12, float entry13, float entry14, float entry15)
  {
    entries[0] = entry0;
    entries[1] = entry1;
    entries[2] = entry2;
    entries[3] = entry3;
    entries[4] = entry4;
    entries[5] = entry5;
    entries[6] = entry6;
    entries[7] = entry7;
    entries[8] = entry8;
    entries[9] = entry9;
    entries[10] = entry10;
    entries[11] = entry11;
    entries[12] = entry12;
    entries[13] = entry13;
    entries[14] = entry14;
    entries[15] = entry15;
  }

  Matrix4 operator* (const Matrix4 &matrix) const
  {
    return Matrix4(entries[0]*matrix.entries[0]+entries[4]*matrix.entries[1]+entries[8]*matrix.entries[2]+entries[12]*matrix.entries[3],
                   entries[1]*matrix.entries[0]+entries[5]*matrix.entries[1]+entries[9]*matrix.entries[2]+entries[13]*matrix.entries[3],
                   entries[2]*matrix.entries[0]+entries[6]*matrix.entries[1]+entries[10]*matrix.entries[2]+entries[14]*matrix.entries[3],
                   entries[3]*matrix.entries[0]+entries[7]*matrix.entries[1]+entries[11]*matrix.entries[2]+entries[15]*matrix.entries[3],
                   entries[0]*matrix.entries[4]+entries[4]*matrix.entries[5]+entries[8]*matrix.entries[6]+entries[12]*matrix.entries[7],
                   entries[1]*matrix.entries[4]+entries[5]*matrix.entries[5]+entries[9]*matrix.entries[6]+entries[13]*matrix.entries[7],
                   entries[2]*matrix.entries[4]+entries[6]*matrix.entries[5]+entries[10]*matrix.entries[6]+entries[14]*matrix.entries[7],
                   entries[3]*matrix.entries[4]+entries[7]*matrix.entries[5]+entries[11]*matrix.entries[6]+entries[15]*matrix.entries[7],
                   entries[0]*matrix.entries[8]+entries[4]*matrix.entries[9]+entries[8]*matrix.entries[10]+entries[12]*matrix.entries[11],
                   entries[1]*matrix.entries[8]+entries[5]*matrix.entries[9]+entries[9]*matrix.entries[10]+entries[13]*matrix.entries[11],
                   entries[2]*matrix.entries[8]+entries[6]*matrix.entries[9]+entries[10]*matrix.entries[10]+entries[14]*matrix.entries[11],
                   entries[3]*matrix.entries[8]+entries[7]*matrix.entries[9]+entries[11]*matrix.entries[10]+entries[15]*matrix.entries[11],
                   entries[0]*matrix.entries[12]+entries[4]*matrix.entries[13]+entries[8]*matrix.entries[14]+entries[12]*matrix.entries[15],
                   entries[1]*matrix.entries[12]+entries[5]*matrix.entries[13]+entries[9]*matrix.entries[14]+entries[13]*matrix.entries[15],
                   entries[2]*matrix.entries[12]+entries[6]*matrix.entries[13]+entries[10]*matrix.entries[14]+entries[14]*matrix.entries[15],
                   entries[3]*matrix.entries[12]+entries[7]*matrix.entries[13]+entries[11]*matrix.entries[14]+entries[15]*matrix.entries[15]);
  }

  Vector3 operator* (const Vector3 &vec) const
  {
    Vector4 result;
    Vector4 v(vec);
    result.x = entries[0]*v.x+entries[4]*v.y+entries[8]*v.z+entries[12]*v.w;
    result.y = entries[1]*v.x+entries[5]*v.y+entries[9]*v.z+entries[13]*v.w;
    result.z = entries[2]*v.x+entries[6]*v.y+entries[10]*v.z+entries[14]*v.w;
    result.w = entries[3]*v.x+entries[7]*v.y+entries[11]*v.z+entries[15]*v.w;
    if((result.w == 0.0f) || (result.w == 1.0f))
    {
      return Vector3(result.x, result.y, result.z);
    }
    else
    {
      const float inv = 1.0f/result.w;
      return Vector3(result.x*inv, result.y*inv, result.z*inv);
    }
  }

  Matrix4 operator* (float scalar) const
  {  
    return Matrix4(entries[0]*scalar, entries[1]*scalar, entries[2]*scalar, entries[3]*scalar,
                   entries[4]*scalar, entries[5]*scalar, entries[6]*scalar, entries[7]*scalar,
                   entries[8]*scalar, entries[9]*scalar, entries[10]*scalar, entries[11]*scalar,
                   entries[12]*scalar, entries[13]*scalar, entries[14]*scalar, entries[15]*scalar);
  }

  Matrix4 operator/ (float scalar) const
  {
    if((scalar == 0.0f) || (scalar == 1.0f))
      return (*this);
    float inv = 1.0f/scalar;
    return (*this)*inv;
  }

  void operator*= (const Matrix4 &rhs)
  {
    (*this) = (*this)*rhs;
  }

  void operator*= (float rhs)
  {
    (*this) = (*this)*rhs;
  }

  void operator/= (float rhs)
  {
    (*this) = (*this)/rhs;
  }

  operator float* () const 
  {
    return (float*) this;
  }

  operator const float* () const  
  {
    return (const float*) this;
  }

  void LoadIdentity()
  {
    memset(entries, 0, 16*sizeof(float));
    entries[0] = 1.0f;
    entries[5] = 1.0f;
    entries[10] = 1.0f;
    entries[15] = 1.0f;
  }

  void LoadZero()
  {
    memset(entries, 0, 16*sizeof(float));
  }

  void Set(float entry0, float entry1, float entry2, float entry3,
           float entry4, float entry5, float entry6, float entry7,
           float entry8, float entry9, float entry10, float entry11,
           float entry12, float entry13, float entry14, float entry15)
  {
    entries[0] = entry0;
    entries[1] = entry1;
    entries[2] = entry2;
    entries[3] = entry3;
    entries[4] = entry4;
    entries[5] = entry5;
    entries[6] = entry6;
    entries[7] = entry7;
    entries[8] = entry8;
    entries[9] = entry9;
    entries[10] = entry10;
    entries[11] = entry11;
    entries[12] = entry12;
    entries[13] = entry13;
    entries[14] = entry14;
    entries[15] = entry15;
  }

  void SetEntry(int index, float value)
  {
    assert((index >= 0) && (index <= 15));
    entries[index] = value;
  }

  float GetEntry(int index) const
  {
    assert((index >= 0) && (index <= 15));
    return entries[index];
  }

  void SetTranslation(const Vector3 &translation)
  {
    entries[12] = translation.x;
    entries[13] = translation.y;
    entries[14] = translation.z;
  }

  void SetRotation(const Vector3 &axis, float angle)
  {
    Vector3 u = axis.GetNormalized();
    const float angleRad = DEG2RAD(angle);
    const float sinAngle = sin(angleRad);
    const float cosAngle = cos(angleRad);
    const float oneMinusCosAngle = 1.0f-cosAngle;
    LoadIdentity();
    entries[0] = (u.x*u.x)+(cosAngle*(1.0f-(u.x*u.x)));
    entries[4] = (u.x*u.y*oneMinusCosAngle)-(sinAngle*u.z);
    entries[8] = (u.x*u.z*oneMinusCosAngle)+(sinAngle*u.y);
    entries[1] = (u.x*u.y*oneMinusCosAngle)+(sinAngle*u.z);
    entries[5] = (u.y*u.y)+(cosAngle*(1.0f-(u.y*u.y)));
    entries[9] = (u.y*u.z)*(oneMinusCosAngle)-(sinAngle*u.x);
    entries[2] = (u.x*u.z)*(oneMinusCosAngle)-(sinAngle*u.y);
    entries[6] = (u.y*u.z)*(oneMinusCosAngle)+(sinAngle*u.x);
    entries[10] = (u.z*u.z)+(cosAngle*(1.0f-(u.z*u.z)));
  }

  void SetRotationX(float angle)
  {
    LoadIdentity();
    const float angleRad = DEG2RAD(angle);
    entries[5] = cos(angleRad);
    entries[6] = sin(angleRad);
    entries[9] = -entries[6];
    entries[10] = entries[5];
  }

  void SetRotationY(float angle)
  {
    LoadIdentity();
    const float angleRad = DEG2RAD(angle);
    entries[0] = cos(angleRad);
    entries[2] = -sin(angleRad);
    entries[8] = -entries[2];
    entries[10] = entries[0];
  }

  void SetRotationZ(float angle)
  {
    LoadIdentity();
    const float angleRad = DEG2RAD(angle);
    entries[0] = cos(angleRad);
    entries[1] = sin(angleRad);
    entries[4] = -entries[1];
    entries[5] = entries[0];
  }

  void SetRotation(const Vector3 &dir)
  {
    Vector3 up, right;
    Vector3 look = dir.GetNormalized();
    if((fabs(look.x) < EPSILON) && (fabs(look.z) < EPSILON))
    {
      if(look.y > 0)
        up.Set(0.0f, 0.0f, -1.0f);
      else
        up.Set(0.0f, 0.0f, 1.0f);
    }
    else
    {
      up.Set(0.0f, 1.0f, 0.0f);
    }

    right = up.CrossProduct(look);
    right.Normalize();
    up = look.CrossProduct(right);  
    up.Normalize();

    entries[0] = right.x;
    entries[1] = right.y;
    entries[2] = right.z;
    entries[3] = 0.0f;

    entries[4] = up.x;
    entries[5] = up.y;
    entries[6] = up.z;
    entries[7] = 0.0f;

    entries[8] = -look.x;
    entries[9] = -look.y;
    entries[10] = -look.z;
    entries[11] = 0.0f;

    entries[12] = 0.0f;
    entries[13] = 0.0f;
    entries[14] = 0.0f;
    entries[15] = 1.0f;
  }

  void SetScale(const Vector3 &scale)
  {
    LoadIdentity();
    entries[0] = scale.x;
    entries[5] = scale.y;
    entries[10] = scale.z;
  }

  void SetPerspective(float fovy, float aspect, float n, float f)
  {
    float left, right, top, bottom;
    fovy = DEG2RAD(fovy);
    top = n*tanf(fovy*0.5f);
    bottom = -top;
    left = aspect*bottom;
    right = aspect*top;
    LoadZero();
    if((left == right) || (top == bottom) || (n == f))
      return;
    entries[0] = (2.0f*n)/(right-left);
    entries[5] = (2.0f*n)/(top-bottom);
    entries[8] = (right+left)/(right-left);
    entries[9] = (top+bottom)/(top-bottom);
    if(f >= 0.0f)
    {
      entries[10] = -(f+n)/(f-n);
      entries[14] = -(2.0f*f*n)/(f-n);
    }
    else
    {
      const float infiniteValue = 0.999f;	
      entries[10] = -infiniteValue;
      entries[14] = -2.0f*n*infiniteValue;
    }
    entries[11] = -1.0f;
  }

  void SetPerspective(const Vector2 &fov, float n, float f)
  {
    float left, right, top, bottom;
    top = n*tanf(DEG2RAD(fov.y)*0.5f);
    bottom = -top;
    right = n*tanf(DEG2RAD(fov.x)*0.5f);
    left = -right;
    LoadZero();
    if((left == right) || (top == bottom) || (n == f))
      return;
    entries[0] = (2.0f*n)/(right-left);
    entries[5] = (2.0f*n)/(top-bottom);
    entries[8] = (right+left)/(right-left);
    entries[9] = (top+bottom)/(top-bottom);
    if(f >= 0.0f)
    {
      entries[10] = -(f+n)/(f-n);
      entries[14] = -(2.0f*f*n)/(f-n);
    }
    else
    {
      const float infiniteValue = 0.999f;	
      entries[10] = -infiniteValue;
      entries[14] = -2.0f*n*infiniteValue;
    }
    entries[11] = -1.0f;
  }

  Matrix4 GetTranspose() const
  {
    return Matrix4(entries[0], entries[4], entries[8], entries[12],
                   entries[1], entries[5], entries[9], entries[13],
                   entries[2], entries[6], entries[10], entries[14],
                   entries[3], entries[7], entries[11], entries[15]);
  }

  void Transpose()
  {
    *this = GetTranspose();
  }

  Matrix4 GetInverseTranspose() const
  {
    Matrix4 result;
    float tmp[12];																								
    tmp[0] = entries[10]*entries[15];
    tmp[1] = entries[11]*entries[14];
    tmp[2] = entries[9]*entries[15];
    tmp[3] = entries[11]*entries[13];
    tmp[4] = entries[9]*entries[14];
    tmp[5] = entries[10]*entries[13];
    tmp[6] = entries[8]*entries[15];
    tmp[7] = entries[11]*entries[12];
    tmp[8] = entries[8]*entries[14];
    tmp[9] = entries[10]*entries[12];
    tmp[10] = entries[8]*entries[13];
    tmp[11] = entries[9]*entries[12];
    result.SetEntry(0, tmp[0]*entries[5]+tmp[3]*entries[6]+tmp[4]*entries[7]
                    -tmp[1]*entries[5]-tmp[2]*entries[6]-tmp[5]*entries[7]);     
    result.SetEntry(1, tmp[1]*entries[4]+tmp[6]*entries[6]+tmp[9]*entries[7]
                    -tmp[0]*entries[4]-tmp[7]*entries[6]-tmp[8]*entries[7]);
    result.SetEntry(2, tmp[2]*entries[4]+tmp[7]*entries[5]+tmp[10]*entries[7]
                    -tmp[3]*entries[4]-tmp[6]*entries[5]-tmp[11]*entries[7]);
    result.SetEntry(3, tmp[5]*entries[4]+tmp[8]*entries[5]+tmp[11]*entries[6]
                    -tmp[4]*entries[4]-tmp[9]*entries[5]-tmp[10]*entries[6]);
    result.SetEntry(4, tmp[1]*entries[1]+tmp[2]*entries[2]+tmp[5]*entries[3]
                    -tmp[0]*entries[1]-tmp[3]*entries[2]-tmp[4]*entries[3]);
    result.SetEntry(5, tmp[0]*entries[0]+tmp[7]*entries[2]+tmp[8]*entries[3]
                    -tmp[1]*entries[0]-tmp[6]*entries[2]-tmp[9]*entries[3]);
    result.SetEntry(6, tmp[3]*entries[0]+tmp[6]*entries[1]+tmp[11]*entries[3]
                    -tmp[2]*entries[0]-tmp[7]*entries[1]-tmp[10]*entries[3]);
    result.SetEntry(7, tmp[4]*entries[0]+tmp[9]*entries[1]+tmp[10]*entries[2]
                    -tmp[5]*entries[0]-tmp[8]*entries[1]-tmp[11]*entries[2]);
    tmp[0] = entries[2]*entries[7];
    tmp[1] = entries[3]*entries[6];
    tmp[2] = entries[1]*entries[7];
    tmp[3] = entries[3]*entries[5];
    tmp[4] = entries[1]*entries[6];
    tmp[5] = entries[2]*entries[5];
    tmp[6] = entries[0]*entries[7];
    tmp[7] = entries[3]*entries[4];
    tmp[8] = entries[0]*entries[6];
    tmp[9] = entries[2]*entries[4];
    tmp[10] = entries[0]*entries[5];
    tmp[11] = entries[1]*entries[4];
    result.SetEntry(8, tmp[0]*entries[13]+tmp[3]*entries[14]+tmp[4]*entries[15]
                    -tmp[1]*entries[13]-tmp[2]*entries[14]-tmp[5]*entries[15]);
    result.SetEntry(9, tmp[1]*entries[12]+tmp[6]*entries[14]+tmp[9]*entries[15]
                    -tmp[0]*entries[12]-tmp[7]*entries[14]-tmp[8]*entries[15]);
    result.SetEntry(10, tmp[2]*entries[12]+tmp[7]*entries[13]+tmp[10]*entries[15]
                    -tmp[3]*entries[12]-tmp[6]*entries[13]-tmp[11]*entries[15]);
    result.SetEntry(11, tmp[5]*entries[12]+tmp[8]*entries[13]+tmp[11]*entries[14]
                    -tmp[4]*entries[12]-tmp[9]*entries[13]-tmp[10]*entries[14]);
    result.SetEntry(12, tmp[2]*entries[10]+tmp[5]*entries[11]+tmp[1]*entries[9]
                    -tmp[4]*entries[11]-tmp[0]*entries[9]-tmp[3]*entries[10]);
    result.SetEntry(13, tmp[8]*entries[11]+tmp[0]*entries[8]+tmp[7]*entries[10]
                    -tmp[6]*entries[10]-tmp[9]*entries[11]-tmp[1]*entries[8]);
    result.SetEntry(14, tmp[6]*entries[9]+tmp[11]*entries[11]+tmp[3]*entries[8]
                    -tmp[10]*entries[11]-tmp[2]*entries[8]-tmp[7]*entries[9]);
    result.SetEntry(15, tmp[10]*entries[10]+tmp[4]*entries[8]+tmp[9]*entries[9]
                    -tmp[8]*entries[9]-tmp[11]*entries[10]-tmp[5]*entries[8]);
    const float det = entries[0]*result.GetEntry(0)+entries[1]*result.GetEntry(1)
                      +entries[2]*result.GetEntry(2)+entries[3]*result.GetEntry(3);
    if(det == 0.0f)
      return Matrix4();
    result = result/det;
    return result;  
  }

  void InvertTranspose()
  {
    *this = GetInverseTranspose();
  }

  void Invert()
  {
    *this = GetInverse();
  }

  Matrix4 GetInverse() const
  {
    Matrix4 result = GetInverseTranspose();
    result.Transpose();
    return result;
  }

  float entries[16];

};

#endif