SET BIN_PATH=bin\win32
IF %PROCESSOR_ARCHITECTURE%==AMD64 SET BIN_PATH=bin\x64

%BIN_PATH%\tiled_forward_shading_demo_Release.exe data/crysponza_bubbles/sponza.obj

pause