#!/bin/sh
CPRO_PATH="/opt/intel/compilers_and_libraries_2017.0.102/mac"
export MKLROOT="${CPRO_PATH}/mkl"
mkl_ld_arch="${CPRO_PATH}/compiler/lib:${MKLROOT}/lib"
mkl_ld_arch="${CPRO_PATH}/tbb/lib:${mkl_ld_arch}"
OLD_DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH}"
export DYLD_LIBRARY_PATH="${mkl_ld_arch}:${OLD_DYLD_LIBRARY_PATH}"
OLD_LIBRARY_PATH="${LIBRARY_PATH}"
export LIBRARY_PATH="${mkl_ld_arch}:${OLD_LIBRARY_PATH}"