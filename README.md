# miniCFD

![](https://github.com/pvcStillInGradSchool/miniCFD/workflows/Build/badge.svg)

## Intention
This repo is a minimum implementation of *Data Structures and Algorithms (DSA)* used in *Computational Fluid Dynamics (CFD)*.

## Build and Test

### Build HDF5

```shell
mkdir HDF5 && cd HDF5
git clone https://github.com/HDFGroup/hdf5.git repo
mkdir build install && cd build
cmake -S ../repo -B . -G Ninja -D CMAKE_BUILD_TYPE=Release -D BUILD_TESTING=OFF -D HDF5_BUILD_TOOLS=OFF -D HDF5_ENABLE_PARALLEL=ON
cmake --build .
cpack -C Release CPackConfig.cmake
cd ../install
../build/HDF5-1.13.1.1-Linux.sh
```

After accepting the license, the script will prompt:

```shell
By default the HDF5 will be installed in:
  "<current directory>/HDF5-1.13.X-Linux"
Do you want to include the subdirectory HDF5-1.13.X-Linux?
Saying no will install in: "<current directory>" [Yn]:
```

Type `n` will get the following directory structure relative to `<current directory>`:

```
install
└── HDF_Group
    └── HDF5
        └── 1.13.X
            ├── bin
            ├── include
            ├── lib
            └── share
                └── cmake
```

Set `MY_HDF5_DIR` to the `share/cmake` directory, which contains some `*.cmake` files.

### Build miniCFD

```shell
git clone https://github.com/pvcStillInGradSchool/miniCFD.git
cd miniCFD
git submodule update --init --recursive
mkdir -p build/Release
cd build/Release
cmake -D CMAKE_BUILD_TYPE=Release -D HDF5_DIR=$MY_HDF5_DIR  -G Ninja -S ../.. -B .  # cmake 3.13.5+
cmake --build .
ctest
```

## Parallel Execution

```shell
cd build/Release/demo/euler
# start a new case
mpirun -n 2 ./shock_tube <filename>.cgns hexa 0.0 0.2 200 10
# restart the old case with a new partition
mpirun -n 4 ./shock_tube ./shock_tube_hexa/shuffled.cgns hexa 0.2 0.4 200 10 200 2
# restart the old case with the old partition
mpirun -n 4 ./shock_tube ./shock_tube_hexa/shuffled.cgns hexa 0.4 0.8 400 10 400 4
```

## Code Style

We follow [*Google's C++ style guide*](http://google.github.io/styleguide/cppguide.html) and use [`cpplint`](https://github.com/cpplint/cpplint) to check our code:

```shell
# Install `cpplint` (one time only):
pip3 install cpplint
# Go to the top source directory:
cd ${MINICFD_SOURCE_DIR}
# Check specific files:
cpplint test/mesh/*.cpp
# Check all source files in `include` and `test`:
cpplint --recursive include test --header=hpp
```
