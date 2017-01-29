#!/bin/sh
g++ -otesttbb -I${TBBROOT}/include testtbb.cpp -L${TBBROOT}/lib -ltbb