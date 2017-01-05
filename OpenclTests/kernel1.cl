#include "commonkern.h"

__kernel void doSome(__global float* buffer)
{
    int i = get_global_id(0);
    float fv = buffer[i];
    buffer[i] = squareIt(fv);
}
