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

double getRandom(double amplitude)
{
    return floor(amplitude * ((double)rand() / (double)RAND_MAX));
}

//-------------------------------------------------------------

void fillTestSystem(const size_t dimension, double* knownX, double* matrix, double* vector)
{
    const double amplitude = 1000;

    for(int i = 0; i < dimension; ++i)
    {
        knownX[i] = getRandom(amplitude);
    }

    for(int row = 0, index = 0; row < dimension; ++row)
    {
        double s = 0;

        for(int col = 0; col < dimension; ++col, ++index)
        {
            double mv = getRandom(amplitude);
            
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

int main()
{
    const int alignment = 32;

    const size_t dim = 4000;    

    const size_t vectorSize = dim * sizeof(double);
    const size_t matrixSize = dim * vectorSize;
    const size_t matrixElementsCount = dim * dim;

    double* fp64Matrix = (double*)mkl_malloc(matrixSize, alignment);
    double* fp64Vector = (double*)mkl_malloc(vectorSize, alignment);
    double* fp64KnownX = (double*)mkl_malloc(vectorSize, alignment);
    double* fp64Solution = (double*)mkl_malloc(vectorSize, alignment);
    double* fp64Residuals = (double*)mkl_malloc(vectorSize, alignment);

    double* fp64LuMatrix = (double*)mkl_malloc(matrixSize, alignment);

    lapack_int* pivotIndices = (lapack_int*)mkl_malloc(dim * sizeof(lapack_int), alignment);

    FILETIME before, after;

    GetSystemTimeAsFileTime(&before);
    srand(before.dwLowDateTime);

    fillTestSystem(dim, fp64KnownX, fp64Matrix, fp64Vector);

    memcpy(fp64LuMatrix, fp64Matrix, matrixSize);
    memcpy(fp64Solution, fp64Vector, vectorSize);

    GetSystemTimeAsFileTime(&before);
        //
    if(!LAPACKE_dgetrf(LAPACK_ROW_MAJOR, dim, dim, fp64LuMatrix, dim, pivotIndices))
    {
        if(!LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N', dim, 1, fp64LuMatrix, dim, pivotIndices, fp64Solution, 1))
        {
            printf("Done.\n");
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

    mkl_free(fp64Residuals);
    mkl_free(fp64Solution);

    mkl_free(pivotIndices);
    mkl_free(fp64LuMatrix);

    mkl_free(fp64KnownX);
    mkl_free(fp64Vector);
    mkl_free(fp64Matrix);

    printf("Bye.\n");    
    
    return 0;
}

//-------------------------------------------------------------