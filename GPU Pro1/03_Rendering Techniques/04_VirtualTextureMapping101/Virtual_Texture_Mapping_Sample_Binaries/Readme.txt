Virtual texture mapping demo binaries
-------------------------------------
Matthäus G. Chajdas <shaderx8@anteru.net>

Overview
~~~~~~~~

This package contains precompiled binaries for the virtual texture mapping
demo. These are directly generated from the source package, using
 
* Visual Studio 2008 SP1 (+9.0.30729.1 SP)
* Boost 1.40 headers
* DirectX August 2009 SDK

They have been tested successfully on Windows Vista x64 bit. The corresponding
dependencies/runtimes are:

* http://www.microsoft.com/downloads/details.aspx?familyid=04AC064B-00D1-474E-B7B1-442D8712D553&displaylang=en[DirectX August 2009 Runtime]
* http://www.microsoft.com/downloads/details.aspx?displaylang=en&FamilyID=2051a0c1-c9b5-4b0a-a8f5-770a549fd78c[Visual C++ 2008 SP1 with ATL Security Update Redistributable]

Make sure that these are correctly installed before running the binaries.

Running
~~~~~~~

In order to run, drop +VTM.exe+ into the +App+ folder from the source, and
run in in that folder. This is explained in more detail in the source package
readme. Make sure that it is directly in the +App+ folder (+App/VTM.exe+),
don't recreate the x86/x64 folder structure below.