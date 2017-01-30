#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <windows.h>

#include <mkl.h>

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

double calcResiduals(size_t dim, double* m, double* x, double* b, double* r)
{
    double ss = 0;

    for(size_t row = 0, index = 0; row < dim; ++row)
    {
        double s = 0;

        for(size_t col = 0; col < dim; ++col, ++index)
        {
            s += m[index] * x[col];
        }

        double e = b[row] - s;

        r[row] = e;

        ss += e * e;
    }

    return ss;
}

//-------------------------------------------------------------

void expandVector(size_t dim, float* fp32v, double* fp64v)
{
    for(size_t i = 0; i < dim; ++i)
    {
        fp64v[i] = (double)(fp32v[i]);
    }
}

void expandMatrix(size_t dim, float* fp32m, double* fp64m)
{
    for(size_t row = 0, index = 0; row < dim; ++row)
    {
        for(size_t col = 0; col < dim; ++col, ++index)
        {
            fp64m[index] = (double)(fp32m[index]);
        }
    }    
}

//-------------------------------------------------------------

int main()
{
    const int alignment = 32;

    const size_t dim = 4000;    

    const size_t fp32VectorSize = dim * sizeof(float);
    const size_t fp32MatrixSize = dim * fp32VectorSize;

    const size_t fp64VectorSize = dim * sizeof(double);
    const size_t fp64MatrixSize = dim * fp64VectorSize;

    float* fp32Matrix = (float*)mkl_malloc(fp32MatrixSize, alignment);
    double* fp64Matrix = (double*)mkl_malloc(fp64MatrixSize, alignment);

    float* fp32Vector = (float*)mkl_malloc(fp32VectorSize, alignment);
    double* fp64Vector = (double*)mkl_malloc(fp64VectorSize, alignment);

    float* fp32KnownX = (float*)mkl_malloc(fp32VectorSize, alignment);

    double* fp64Solution = (double*)mkl_malloc(fp64VectorSize, alignment);
    double* fp64Residuals = (double*)mkl_malloc(fp64VectorSize, alignment);
    double* fp64Residuals2 = (double*)mkl_malloc(fp64VectorSize, alignment);

    float* fp32LuMatrix = (float*)mkl_malloc(fp32MatrixSize, alignment);
    double* fp64LuMatrix = (double*)mkl_malloc(fp64MatrixSize, alignment);

    lapack_int* pivotIndices = (lapack_int*)mkl_malloc(dim * sizeof(lapack_int), alignment);

    FILETIME before, after;

    GetSystemTimeAsFileTime(&before);
    srand(before.dwLowDateTime);

    fillTestSystem(dim, fp32KnownX, fp32Matrix, fp32Vector);

    expandVector(dim, fp32Vector, fp64Vector);
    expandMatrix(dim, fp32Matrix, fp64Matrix);

    memcpy(fp32LuMatrix, fp32Matrix, fp32MatrixSize);
    memcpy(fp64Solution, fp64Vector, fp64VectorSize);

    GetSystemTimeAsFileTime(&before);
        //
    if(!LAPACKE_sgetrf(LAPACK_ROW_MAJOR, dim, dim, fp32LuMatrix, dim, pivotIndices))
    {
        expandMatrix(dim, fp32LuMatrix, fp64LuMatrix);

        if(!LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N', dim, 1, fp64LuMatrix, dim, pivotIndices, fp64Solution, 1))
        {
            if(!LAPACKE_dgerfs(LAPACK_ROW_MAJOR, 'N', dim, 1, fp64Matrix, dim, fp64LuMatrix, dim, pivotIndices, fp64Vector, 1, fp64Solution, 1, fp64Residuals, fp64Residuals2))
            {
                printf("Done.\n");
            }
            else
            {
                printf("Error 3\n");    
            }            
        }
        else
        {
            printf("Error 2\n");
        }
    }
    else
    {
        printf("Error 1\n");
    }
        //
    GetSystemTimeAsFileTime(&after);

    printf("Execution time: %u ms.\n", timeDifference(&before, &after) / 1000);

    printf("%.7e\n", calcResiduals(dim, fp64Matrix, fp64Solution, fp64Vector, fp64Residuals));

cleanup:

    mkl_free(fp64Residuals2);
    mkl_free(fp64Residuals);
    mkl_free(fp64Solution);

    mkl_free(pivotIndices);

    mkl_free(fp64LuMatrix);
    mkl_free(fp32LuMatrix);

    mkl_free(fp32KnownX);

    mkl_free(fp64Vector);
    mkl_free(fp32Vector);
    mkl_free(fp64Matrix);
    mkl_free(fp32Matrix);

    printf("Bye.\n");    
    
    return 0;
}

//-------------------------------------------------------------