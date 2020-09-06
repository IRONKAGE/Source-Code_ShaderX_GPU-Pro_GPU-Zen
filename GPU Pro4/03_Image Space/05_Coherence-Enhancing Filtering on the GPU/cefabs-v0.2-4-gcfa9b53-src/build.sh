#!/bin/sh
cmake -E make_directory build
cd build
if [ `uname` == "Darwin" ]; then
	cmake -G Xcode -DCMAKE_OSX_DEPLOYMENT_TARGET=10.6 ..
	xcodebuild -configuration Release
else
	cmake ..
fi
make
cpack --config CPackConfig.cmake
