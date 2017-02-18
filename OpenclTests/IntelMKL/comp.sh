#!/bin/sh
g++ -otmkl  -DMKL_ILP64 -std=c++11 -mavx -Ofast -march=haswell -funroll-loops -I${MKLROOT}/include tmkl.cpp ${MKLROOT}/lib/libmkl_intel_ilp64.a ${MKLROOT}/lib/libmkl_tbb_thread.a ${MKLROOT}/lib/libmkl_core.a -ltbb -lstdc++ -lpthread -lm -ldl -L/opt/intel/tbb/lib
#g++ -otmkld  -DMKL_ILP64 -std=c++11 -mavx -Ofast -march=haswell -funroll-loops -I${MKLROOT}/include tmkld.cpp ${MKLROOT}/lib/libmkl_intel_ilp64.a ${MKLROOT}/lib/libmkl_tbb_thread.a ${MKLROOT}/lib/libmkl_core.a -ltbb -lstdc++ -lpthread -lm -ldl -L/opt/intel/tbb/lib
