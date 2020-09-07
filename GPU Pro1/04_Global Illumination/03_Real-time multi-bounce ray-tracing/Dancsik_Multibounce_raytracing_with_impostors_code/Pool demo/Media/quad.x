xof 0303txt 0032
template Frame {
 <3d82ab46-62da-11cf-ab39-0020af71e433>
 [...]
}

template Matrix4x4 {
 <f6f23f45-7686-11cf-8f52-0040333594a3>
 array FLOAT matrix[16];
}

template FrameTransformMatrix {
 <f6f23f41-7686-11cf-8f52-0040333594a3>
 Matrix4x4 frameMatrix;
}

template Vector {
 <3d82ab5e-62da-11cf-ab39-0020af71e433>
 FLOAT x;
 FLOAT y;
 FLOAT z;
}

template MeshFace {
 <3d82ab5f-62da-11cf-ab39-0020af71e433>
 DWORD nFaceVertexIndices;
 array DWORD faceVertexIndices[nFaceVertexIndices];
}

template Mesh {
 <3d82ab44-62da-11cf-ab39-0020af71e433>
 DWORD nVertices;
 array Vector vertices[nVertices];
 DWORD nFaces;
 array MeshFace faces[nFaces];
 [...]
}

template MeshNormals {
 <f6f23f43-7686-11cf-8f52-0040333594a3>
 DWORD nNormals;
 array Vector normals[nNormals];
 DWORD nFaceNormals;
 array MeshFace faceNormals[nFaceNormals];
}

template Coords2d {
 <f6f23f44-7686-11cf-8f52-0040333594a3>
 FLOAT u;
 FLOAT v;
}

template MeshTextureCoords {
 <f6f23f40-7686-11cf-8f52-0040333594a3>
 DWORD nTextureCoords;
 array Coords2d textureCoords[nTextureCoords];
}

template ColorRGBA {
 <35ff44e0-6c7c-11cf-8f52-0040333594a3>
 FLOAT red;
 FLOAT green;
 FLOAT blue;
 FLOAT alpha;
}

template IndexedColor {
 <1630b820-7842-11cf-8f52-0040333594a3>
 DWORD index;
 ColorRGBA indexColor;
}

template MeshVertexColors {
 <1630b821-7842-11cf-8f52-0040333594a3>
 DWORD nVertexColors;
 array IndexedColor vertexColors[nVertexColors];
}

template VertexElement {
 <f752461c-1e23-48f6-b9f8-8350850f336f>
 DWORD Type;
 DWORD Method;
 DWORD Usage;
 DWORD UsageIndex;
}

template DeclData {
 <bf22e553-292c-4781-9fea-62bd554bdd93>
 DWORD nElements;
 array VertexElement Elements[nElements];
 DWORD nDWords;
 array DWORD data[nDWords];
}


Frame DXCC_ROOT {
 

 FrameTransformMatrix {
  1.000000,0.000000,0.000000,0.000000,0.000000,1.000000,0.000000,0.000000,0.000000,0.000000,1.000000,0.000000,0.000000,0.000000,0.000000,1.000000;;
 }

 Frame persp {
  

  FrameTransformMatrix {
   0.952129,0.000000,0.305695,0.000000,-0.052759,0.984994,0.164327,0.000000,-0.301108,-0.172588,0.937842,0.000000,0.967594,0.554604,-3.013704,1.000000;;
  }

  Frame perspShape {
   

   FrameTransformMatrix {
    1.000000,0.000000,0.000000,0.000000,0.000000,1.000000,0.000000,0.000000,0.000000,0.000000,1.000000,0.000000,0.000000,0.000000,0.000000,1.000000;;
   }
  }
 }

 Frame top {
  

  FrameTransformMatrix {
   1.000000,0.000000,0.000000,0.000000,0.000000,0.000000,1.000000,0.000000,0.000000,-1.000000,0.000000,0.000000,0.000000,100.099998,0.000000,1.000000;;
  }

  Frame topShape {
   

   FrameTransformMatrix {
    1.000000,0.000000,0.000000,0.000000,0.000000,1.000000,0.000000,0.000000,0.000000,0.000000,1.000000,0.000000,0.000000,0.000000,0.000000,1.000000;;
   }
  }
 }

 Frame front {
  

  FrameTransformMatrix {
   1.000000,0.000000,0.000000,0.000000,0.000000,1.000000,0.000000,0.000000,0.000000,0.000000,1.000000,0.000000,0.000000,0.000000,-100.099998,1.000000;;
  }

  Frame frontShape {
   

   FrameTransformMatrix {
    1.000000,0.000000,0.000000,0.000000,0.000000,1.000000,0.000000,0.000000,0.000000,0.000000,1.000000,0.000000,0.000000,0.000000,0.000000,1.000000;;
   }
  }
 }

 Frame side {
  

  FrameTransformMatrix {
   0.000000,0.000000,1.000000,0.000000,0.000000,1.000000,0.000000,0.000000,-1.000000,0.000000,0.000000,0.000000,100.099998,0.000000,0.000000,1.000000;;
  }

  Frame sideShape {
   

   FrameTransformMatrix {
    1.000000,0.000000,0.000000,0.000000,0.000000,1.000000,0.000000,0.000000,0.000000,0.000000,1.000000,0.000000,0.000000,0.000000,0.000000,1.000000;;
   }
  }
 }

 Frame pPlane1 {
  

  FrameTransformMatrix {
   1.000000,0.000000,0.000000,0.000000,0.000000,1.000000,0.000000,0.000000,0.000000,0.000000,1.000000,0.000000,0.000000,0.000000,0.000000,1.000000;;
  }

  Frame pPlaneShape1 {
   

   FrameTransformMatrix {
    1.000000,0.000000,0.000000,0.000000,0.000000,1.000000,0.000000,0.000000,0.000000,0.000000,1.000000,0.000000,0.000000,0.000000,0.000000,1.000000;;
   }

   Mesh pPlaneShape1_Mesh {
    4;
    -1.000000;-1.000000;0.000000;,
    -1.000000;1.000000;0.000000;,
    1.000000;-1.000000;0.000000;,
    1.000000;1.000000;0.000000;;
    2;
    3;0,1,2;,
    3;2,1,3;;

    MeshNormals {
     4;
     0.000000;0.000000;-1.000000;,
     0.000000;0.000000;-1.000000;,
     0.000000;0.000000;-1.000000;,
     0.000000;0.000000;-1.000000;;
     2;
     3;0,1,2;,
     3;2,1,3;;
    }

    MeshTextureCoords {
     4;
     0.000000;1.000000;,
     0.000000;0.000000;,
     1.000000;1.000000;,
     1.000000;0.000000;;
    }

    MeshVertexColors {
     4;
     0;0.000000;0.000000;0.000000;0.000000;;,
     1;0.000000;0.000000;0.000000;0.000000;;,
     2;0.000000;0.000000;0.000000;0.000000;;,
     3;0.000000;0.000000;0.000000;0.000000;;;
    }

    DeclData {
     2;
     2;0;6;0;,
     2;0;7;0;;
     24;
     0,
     3212836864,
     0,
     3212836864,
     2995291556,
     0,
     839419304,
     3212836864,
     0,
     3212836864,
     2999765112,
     0,
     839419304,
     3212836864,
     0,
     3212836864,
     2999765112,
     0,
     3007036717,
     3212836864,
     0,
     3212836864,
     867941678,
     0;
    }
   }
  }
 }
}