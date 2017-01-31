#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <windows.h>

#include "solver7sse.h"

//-------------------------------------------------------------

unsigned int timeDifference(LPFILETIME before, LPFILETIME after)
{
    ULARGE_INTEGER beforeQ, afterQ;

    beforeQ.u.LowPart = before->dwLowDateTime;
    beforeQ.u.HighPart = before->dwHighDateTime;

    afterQ.u.LowPart = after->dwLowDateTime;
    afterQ.u.HighPart = after->dwHighDateTime;
    
    ULONGLONG diff = (afterQ.QuadPart - beforeQ.QuadPart) / 10ULL;

    return (unsigned int)diff;
}

//-------------------------------------------------------------

float getRandom(float amplitude)
{
    return floor(amplitude * ((float)rand() / (float)RAND_MAX));
}

//-------------------------------------------------------------

void fillTestSystem(const size_t dimension, float* knownX, float* matrix, float* vector)
{
    const float amplitude = 1000;

    for(int i = 0; i < dimension; ++i)
    {
        knownX[i] = getRandom(amplitude);
    }

    for(int row = 0, index = 0; row < dimension; ++row)
    {
        float s = 0;

        for(int col = 0; col < dimension; ++col, ++index)
        {
            float mv = getRandom(amplitude);
            
            matrix[index] = mv;

            s += mv * knownX[col];
        }

        vector[row] = s;
    }
}

//-------------------------------------------------------------

int main()
{
    const size_t dim = 4000;    
    const size_t matrixElementsCount = dim * dim;

    float* fp32Matrix = (float*)malloc(matrixElementsCount * sizeof(float));
    float* fp32Vector = (float*)malloc(dim * sizeof(float));
    float* fp32KnownX = (float*)malloc(dim * sizeof(float));

    FILETIME before, after;

    GetSystemTimeAsFileTime(&before);
    srand(before.dwLowDateTime);

    Solver slv;

    if(!slv.Init(dim))
    {
        printf("Error Solver init\n");
        goto cleanup;   
    }

    fillTestSystem(dim, fp32KnownX, fp32Matrix, fp32Vector);

    slv.useMatrix(fp32Matrix);
    slv.useVector(fp32Vector);

    printf("%lu %lu ", slv.expandedDimension, slv.sseBlocksCount);

    printf("cpus %lu\n", slv.ncpu);

    GetSystemTimeAsFileTime(&before);
    slv.Iterate(3);
    GetSystemTimeAsFileTime(&after);

    printf("Execution time: %u ms.\n", timeDifference(&before, &after) / 1000);

    printf("%.7e\n", slv.CalcResiduals());

cleanup:

    slv.Dispose();
    
    free(fp32KnownX);
    free(fp32Vector);
    free(fp32Matrix);

    printf("Bye.\n");    
    
    return 0;
}

//-------------------------------------------------------------
