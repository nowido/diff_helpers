#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <sys/time.h>
#include <unistd.h>

#include "ocl_playground.h"
#include "solver6.h"

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

double dotProductFp64(const size_t dimension, double* x1, double* x2)
{
    double s = 0;

    for(int i = 0; i < dimension; ++i)
    {
        s += x1[i] * x2[i];
    }

    return s;
}

//-------------------------------------------------------------

void calcResidualsFp64(const size_t dimension, double* x, double* matrix, double* vector, double* residuals)
{
    for(int row = 0; row < dimension; ++row)
    {
        residuals[row] = vector[row] - dotProductFp64(dimension, matrix + row * dimension, x);
    }
}

//-------------------------------------------------------------

void addVectorFp64(const size_t dimension, double* vectorDest, double* vectorSrc)
{
    for(int i = 0; i < dimension; ++i)
    {
        vectorDest[i] += vectorSrc[i];
    }
}

//-------------------------------------------------------------

void averageVectorsFp64(const size_t dimension, double* vectorDest, double* vectorSrc)
{
    for(int i = 0; i < dimension; ++i)
    {
        double v1 = vectorDest[i];
        double v2 = vectorSrc[i];
        vectorDest[i] = (v1 + v2) / 2.0;
    }
}

//-------------------------------------------------------------

void reverseMatrixInPlaceFp32(const size_t dimension, float* matrix)
{
    for(int col = 0, back = dimension - 1; col < back; ++col, --back)
    {
        for(int row = 0, index1 = col, index2 = back; row < dimension; ++row, index1 += dimension, index2 += dimension)
        {
            float t = matrix[index1];
            matrix[index1] = matrix[index2];
            matrix[index2] = t;
        }    
    }
}

//-------------------------------------------------------------

void reverseVectorInPlaceFp64(const size_t dimension, double* vector)
{
    for(int i = 0, back = dimension - 1; i < back; ++i, --back)
    {
        double t = vector[i];
        vector[i] = vector[back];
        vector[back] = t;
    }
}

//-------------------------------------------------------------

int main()
{
    const size_t dim = 1000;    
    const size_t matrixElementsCount = dim * dim;
    const size_t matrixBytesCountFp32 = matrixElementsCount * sizeof(float);

    float* matrixFp32 = (float*)malloc(matrixBytesCountFp32);

    float* vectorFp32 = (float*)malloc(dim * sizeof(float));

    float* knownX = (float*)malloc(dim * sizeof(float));

    double* residuals = (double*)malloc(dim * sizeof(double));

    struct timeval before, after;

    gettimeofday(&before, NULL);
    srand(before.tv_usec);

    fillTestSystem(dim, knownX, matrixFp32, vectorFp32);

        //

    ClPlayground cl;    

    Solver slv1;
    Solver slv2;

    if(!cl.Init())
    {
        printf("Error CL init\n");        

        goto cleanup;        
    }
    
    if(!(slv1.Init(dim, &cl, cl_notify) && slv2.Init(dim, &cl, cl_notify)))
    {
        printf("Error Solver init\n");

        printf("%s\n", cl.buildLog);

        goto cleanup;                
    }

    slv1.useMatrix(matrixFp32);
    slv1.useVector(vectorFp32);

    reverseMatrixInPlaceFp32(dim, matrixFp32);

    slv2.useMatrix(matrixFp32);
    slv2.useVector(vectorFp32);

    gettimeofday(&before, NULL);    
    //slv1.Solve();
    //slv2.Solve();
    slv1.Iterate(2);
    slv2.Iterate(2);
        
    //calcResidualsFp64(dim, slv.solution, slv.matrixFp64, slv.vectorFp64, residuals);

    //printf("residuals dp: %f\n", dotProductFp64(dim, residuals, residuals));

    printf("residuals1 dp: %15e\n", slv1.CalcResiduals());
    printf("residuals2 dp: %15e\n", slv2.CalcResiduals());

    reverseVectorInPlaceFp64(dim, slv2.solution);
    averageVectorsFp64(dim, slv1.solution, slv2.solution);

        for(size_t i = 0; i < 1; ++i)
        {
            slv1.CalcResiduals();

            slv1.useLupToSolve(slv1.iterativeSolution, slv1.residuals);
            addVectorFp64(dim, slv1.solution, slv1.iterativeSolution);
        }

    gettimeofday(&after, NULL);
    printf("Execution time: %u ms.\n", timeDifference(&before, &after) / 1000);

    printf("residuals av dp: %15e\n", slv1.CalcResiduals());

cleanup:

    slv1.Dispose();
    slv2.Dispose();
    cl.Dispose();

    free(knownX);
    free(vectorFp32);
    free(matrixFp32);
    free(residuals);

    printf("Bye.\n");    
    
    return 0;
}

//-------------------------------------------------------------