#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <windows.h>

#include "solver7sse.h"

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

int CompareResources(const size_t dimension, float* matrix1, size_t stride1, float* matrix2, size_t stride2)
{
    int count = 0;

    float* src = matrix1;
    float* dest = matrix2;

    for(size_t row = 0; row < dimension; ++row, src += stride1, dest += stride2)
    {
        for(size_t col = 0; col < dimension; ++col)
        {
            if(src[col] != dest[col])
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

    Solver slv;

    if(!slv.Init(dim))
    {
        printf("Error Solver init\n");
        goto cleanup;   
    }

    fillTestSystem(dim, fp32KnownX, fp32Matrix, fp32Vector);

    slv.useMatrix(fp32Matrix);

    int c = CompareResources(dim, fp32Matrix, dim, slv.fp32Matrix, slv.fp32VectorStride / sizeof(float));

    printf("%d %d ", slv.actualDimension, slv.sseBlocksCount);

    printf("%d\n", c);

cleanup:

    free(fp32KnownX);
    free(fp32Vector);
    free(fp32Matrix);

    printf("Bye.\n");    
    
    return 0;
}

//-------------------------------------------------------------