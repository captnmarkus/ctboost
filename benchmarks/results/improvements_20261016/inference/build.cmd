@echo off
set "PATH=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64;C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64;%PATH%"
set "INCLUDE=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\include;C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\ucrt;C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\shared;C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\um;C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\winrt"
set "LIB=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\lib\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\ucrt\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\um\x64"
cmake -S C:\apps\ctboost -B C:\apps\ctboost\.tmp\improvements-20261016\build-release -G Ninja -DCMAKE_BUILD_TYPE=Release -DCTBOOST_ENABLE_CUDA=OFF -DCTBOOST_BUILD_NATIVE_TESTS=ON -DPython_EXECUTABLE=C:/Users/captn/miniconda3/python.exe -Dpybind11_DIR=C:/Users/captn/miniconda3/Lib/site-packages/pybind11/share/cmake/pybind11
if errorlevel 1 exit /b %errorlevel%
cmake --build C:\apps\ctboost\.tmp\improvements-20261016\build-release --config Release --parallel 2

