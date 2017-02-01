#!/bin/sh
#g++ -std=c++11 -otesttbb -I${TBBROOT}/include testtbb.cpp -L${TBBROOT}/lib -ltbb
g++ -std=c++11 -m64 -Ofast -march=native -funroll-loops -otesttbb -I${TBBROOT}/include testtbb.cpp -L${TBBROOT}/lib -ltbb
