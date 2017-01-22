#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <sys/time.h>
#include <unistd.h>

#include "solver7sse.h"

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

int CompareResources1(const size_t dimension, float* matrix1, size_t stride1, float* matrix2, size_t stride2)
{
    int count = 0;

    for(size_t row = 0; row < dimension; ++row)
    {
        for(size_t col = 0; col < dimension; ++col)
        {
            size_t srcIndex = row * stride1 + col;
            size_t transpIndex = col * stride2 + row;

            if(matrix1[srcIndex] != matrix2[transpIndex])
            {
                ++count;
            }
        }
    }

    return count;
}

int CompareResources2(const size_t dimension, float* matrix1, size_t stride1, double* matrix2, size_t stride2)
{
    int count = 0;

    for(size_t row = 0; row < dimension; ++row)
    {
        for(size_t col = 0; col < dimension; ++col)
        {
            size_t srcIndex = row * stride1 + col;
            size_t transpIndex = col * stride2 + row;

            if((double)(matrix1[srcIndex]) != matrix2[transpIndex])
            {
                ++count;
            }
        }
    }

    return count;
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

    printf("%lu %lu ", slv.expandedDimension, slv.sseBlocksCount);

    {
        int c1 = CompareResources1(dim, fp32Matrix, dim, slv.fp32Matrix, slv.fp32VectorStride / sizeof(float));
        int c2 = CompareResources2(dim, fp32Matrix, dim, slv.fp64Matrix, slv.fp64VectorStride / sizeof(double));

        printf("%d %d\n", c1, c2);
    }

    printf("cpus %lu\n", slv.ncpu);

    gettimeofday(&before, NULL);
    for(int i = 0; i < 10; ++i)
    slv.Solve();
    gettimeofday(&after, NULL);

    printf("Execution time: %u ms.\n", timeDifference(&before, &after) / 1000);

cleanup:

    free(fp32KnownX);
    free(fp32Vector);
    free(fp32Matrix);

    printf("Bye.\n");    
    
    return 0;
}

//-------------------------------------------------------------
