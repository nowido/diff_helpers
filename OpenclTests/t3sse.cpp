#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <sys/time.h>
#include <unistd.h>

//#include "solver9blavx.h"
#include "solver9blasavx.h"
//#include "solver9blas.h"

//-------------------------------------------------------------

unsigned int timeDifference(struct timeval *before, struct timeval *after)
{
    return ((after->tv_sec * 1000000 + after->tv_usec) - (before->tv_sec * 1000000 + before->tv_usec));
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

    struct timeval before, after;

    gettimeofday(&before, NULL);
    srand(before.tv_usec);

    Solver slv;

    if(!slv.Init(dim))
    {
        printf("Error Solver init\n");
        goto cleanup;   
    }

    fillTestSystem(dim, fp32KnownX, fp32Matrix, fp32Vector);

    slv.useMatrix(fp32Matrix);
    slv.useVector(fp32Vector);

    printf("%lu\n", slv.expandedDimension);

    gettimeofday(&before, NULL);
    
    //slv.Solve();

    slv.Iterate(3);
    gettimeofday(&after, NULL);

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
