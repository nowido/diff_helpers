#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <windows.h>

#include "ocl_playground.h"
#include "solver6comb1g.h"

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
    const size_t dim = 4000;    
    const size_t matrixElementsCount = dim * dim;
    const size_t matrixBytesCountFp32 = matrixElementsCount * sizeof(float);

    float* matrixFp32 = (float*)malloc(matrixBytesCountFp32);

    float* vectorFp32 = (float*)malloc(dim * sizeof(float));

    float* knownX = (float*)malloc(dim * sizeof(float));

    double* residuals = (double*)malloc(dim * sizeof(double));

    FILETIME before, after;

    GetSystemTimeAsFileTime(&before);
    srand(before.dwLowDateTime);

    fillTestSystem(dim, knownX, matrixFp32, vectorFp32);

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

    slv.useMatrix(matrixFp32);
    slv.useVector(vectorFp32);

    GetSystemTimeAsFileTime(&before);  
    //slv.Solve();
    slv.Iterate(2);

    printf("residuals1 dp: %15e\n", slv.CalcResiduals(slv.matrix1_Fp64, slv.solution1, slv.residuals1));
    printf("residuals2 dp: %15e\n", slv.CalcResiduals(slv.matrix2_Fp64, slv.solution2, slv.residuals2));

    reverseVectorInPlaceFp64(dim, slv.solution2);
    averageVectorsFp64(dim, slv.solution1, slv.solution2);

        for(size_t i = 0; i < 1; ++i)
        {
            slv.CalcResiduals(slv.matrix1_Fp64, slv.solution1, slv.residuals1);

            slv.useLupToSolve(slv.permutation1, slv.lu1_Fp64, slv.iterativeSolution1, slv.residuals1);
            addVectorFp64(dim, slv.solution1, slv.iterativeSolution1);
        }

    GetSystemTimeAsFileTime(&after);
    printf("Execution time: %u ms.\n", timeDifference(&before, &after) / 1000);

    printf("residuals av dp: %15e\n", slv.CalcResiduals(slv.matrix1_Fp64, slv.solution1, slv.vectorFp64));

cleanup:

    slv.Dispose();    
    cl.Dispose();

    free(knownX);
    free(vectorFp32);
    free(matrixFp32);
    free(residuals);

    printf("Bye.\n");    
    
    return 0;
}

//-------------------------------------------------------------