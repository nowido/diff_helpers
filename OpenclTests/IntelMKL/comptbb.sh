#!/bin/sh
g++ -std=c++11 -otesttbb -I${TBBROOT}/include testtbb.cpp -L${TBBROOT}/lib -ltbb
