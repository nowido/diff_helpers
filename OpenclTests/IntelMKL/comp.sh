#!/bin/sh
g++ -otmkld  -DMKL_ILP64 -m64 -I${MKLROOT}/include tmkld.cpp ${MKLROOT}/lib/libmkl_intel_ilp64.a ${MKLROOT}/lib/libmkl_tbb_thread.a ${MKLROOT}/lib/libmkl_core.a -ltbb -lstdc++ -lpthread -lm -ldl -L/opt/intel/tbb/lib