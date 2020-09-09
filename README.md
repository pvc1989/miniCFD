# miniCFD

## Intention
This repo is a minimum implementation of *Data Structures and Algorithms (DSA)* used in *Computational Fluid Dynamics (CFD)*.

## Build
```shell
git clone https://github.com/pvcStillInGradSchool/miniCFD.git
git submodule update --init --recursive
cd miniCFD
mkdir -p build/Debug
cd build/Debug
cmake -D CMAKE_BUILD_TYPE=Debug -G Ninja -S ../.. -B .  # cmake 3.13.5+
cmake --build .
ctest
mkdir result
./demo/euler/tube sod tube.vtk 0.0 0.5 500 5
./demo/euler/box  sod  box.vtk 0.0 1.0 800 5
```
