#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <sys/time.h>
#include <unistd.h>

#include "ocl_playground.h"
#include "solverflgr.h"

//-------------------------------------------------------------

unsigned int timeDifference(struct timeval *before, struct timeval *after)
{
    return ((after->tv_sec * 1000000 + after->tv_usec) - (before->tv_sec * 1000000 + before->tv_usec));
}

//-------------------------------------------------------------

void CL_CALLBACK cl_notify
        (
            const char *errinfo, 
            const void *private_info, 
            size_t cb, 
            void *user_data
        )
{
    printf("CL ERROR: %s\n", errinfo);
}

//-------------------------------------------------------------

float getRandom(float amplitude)
{
    return floor(amplitude * ((float)rand() / (float)RAND_MAX));
}

//-------------------------------------------------------------

void fillTestSystem(const size_t dimension, float* knownX, float* matrix, float* vector)
{
    const float amplitude = 10;

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

            float kx = knownX[col];

            s += mv * kx;
        }

        vector[row] = s;
    }
}

//-------------------------------------------------------------

int main()
{
    const size_t dim = 1000;

    float* matrix = (float*)malloc(dim * dim * sizeof(float));
    float* vector = (float*)malloc(dim * sizeof(float));
    float* knownX = (float*)malloc(dim * sizeof(float));

        //

    struct timeval before, after;

    gettimeofday(&before, NULL);
    srand(before.tv_usec);

    fillTestSystem(dim, knownX, matrix, vector);

        //

    ClPlayground cl;    

    Solver slv;

    if(!cl.Init())
    {
        printf("Error CL init\n");        

        goto cleanup;        
    }

    if(!slv.Init(dim, &cl, cl_notify))
    {
        printf("Error Solver init\n");

        printf("%s\n", cl.buildLog);

        goto cleanup;                
    }

    slv.useMatrix(matrix);
    slv.useVector(vector);

    gettimeofday(&before, NULL);
    slv.Solve();
    //slv.Iterate(3);
    gettimeofday(&after, NULL);

    printf("Execution time: %u ms.\n", timeDifference(&before, &after) / 1000);

    slv.CalcResiduals();

    printf("Residuals squares sum: %f\n", slv.residualsSquaresSum);

cleanup:

    slv.Dispose();
    cl.Dispose();

    free(knownX);
    free(vector);
    free(matrix);

    printf("Bye.\n");    
    return 0;        
}

//-------------------------------------------------------------