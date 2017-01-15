#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <sys/time.h>
#include <unistd.h>

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

float dotProductFp32(const size_t dimension, float* x1, float* x2)
{
    float s = 0;

    for(int i = 0; i < dimension; ++i)
    {
        s += x1[i] * x2[i];
    }

    return s;
}

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

void calcResidualsFp32(const size_t dimension, float* x, float* matrix, float* vector, float* residuals)
{
    for(int row = 0; row < dimension; ++row)
    {
        residuals[row] = vector[row] - dotProductFp32(dimension, matrix + row * dimension, x);
    }
}

void calcResidualsFp64(const size_t dimension, double* x, double* matrix, double* vector, double* residuals)
{
    for(int row = 0; row < dimension; ++row)
    {
        residuals[row] = vector[row] - dotProductFp64(dimension, matrix + row * dimension, x);
    }
}

//-------------------------------------------------------------

int findPivotRow(const size_t dimension, float* matrix, int diagIndex)
{
    int prow = -1;

    float maxValue = 0;

    for(int i = diagIndex, index = diagIndex * dimension + diagIndex; i < dimension; ++i, index += dimension)
    {
        float mv = fabs(matrix[index]);

        if(mv > maxValue)
        {
            maxValue = mv;
            prow = i;
        }
    }

    return prow;
}

//-------------------------------------------------------------

void swapRows(const size_t dimension, float* matrix, int r1, int r2)
{
    for(int i = 0, index1 = r1 * dimension, index2 = r2 * dimension; i < dimension; ++i, ++index1, ++index2)
    {
        float tmp = matrix[index1];
        matrix[index1] = matrix[index2];
        matrix[index2] = tmp;
    }
}

//-------------------------------------------------------------

bool lupFactorization(const size_t dimension, float* matrix, int* p)
{
    for(int step = 0, leadIndex = 0; step < dimension; ++step, leadIndex += dimension)
    {
        int pivotRow = findPivotRow(dimension, matrix, step);
        
        if(pivotRow < 0)
        {
            return false;
        }

        p[step] = pivotRow;

        if(pivotRow != step)
        {
            swapRows(dimension, matrix, step, pivotRow);            
        }

        float ld = matrix[leadIndex + step];

        for(int row = step + 1, index = row * dimension; row < dimension; ++row, index += dimension)
        {
            float lv = (matrix[index + step] /= ld);

            for(int col = step + 1; col < dimension; ++col)
            {
                matrix[index + col] -= matrix[leadIndex + col] * lv;    
            }            
        }
    }

    return true;
}

//-------------------------------------------------------------

void solutionFp32(const size_t dimension, float* lu, int* p, float* vector, float* x)
{
    memcpy(x, vector, dimension * sizeof(float));

        // permute right-hand part

    for(int row = 0; row < dimension; ++row)
    {
        int permutIndex = p[row];

        if(permutIndex != row)
        {
            float tmp = x[permutIndex];
            x[permutIndex] = x[row];        
            x[row] = tmp;
        }
    }
        // Ly = Pb (in place)
    
    for(int row = 1, index = dimension; row < dimension; ++row, index += dimension)
    {
        float s = 0;

        for(int col = 0; col < row; ++col)
        {
            s += lu[index + col] * x[col];
        }

        x[row] -= s;
    }

        // Ux = y

    for(int row = dimension - 1, index = row * dimension; row >= 0; --row, index -= dimension)
    {
        float s = 0;

        for(int col = row + 1; col < dimension; ++col)
        {
            s += lu[index + col] * x[col];
        }

        x[row] -= s;
        x[row] /= lu[index + row];        
    }
}

void solutionFp64(const size_t dimension, double* lu, int* p, double* vector, double* x)
{
    memcpy(x, vector, dimension * sizeof(double));

        // permute right-hand part

    for(int row = 0; row < dimension; ++row)
    {
        int permutIndex = p[row];

        if(permutIndex != row)
        {
            double tmp = x[permutIndex];
            x[permutIndex] = x[row];        
            x[row] = tmp;
        }
    }
        // Ly = Pb (in place)
    
    for(int row = 1, index = dimension; row < dimension; ++row, index += dimension)
    {
        double s = 0;

        for(int col = 0; col < row; ++col)
        {
            s += lu[index + col] * x[col];
        }

        x[row] -= s;
    }

        // Ux = y

    for(int row = dimension - 1, index = row * dimension; row >= 0; --row, index -= dimension)
    {
        double s = 0;

        for(int col = row + 1; col < dimension; ++col)
        {
            s += lu[index + col] * x[col];
        }

        x[row] -= s;
        x[row] /= lu[index + row];        
    }
}

//-------------------------------------------------------------

void addVectorFp32(const size_t dimension, float* vectorDest, float* vectorSrc)
{
    for(int i = 0; i < dimension; ++i)
    {
        vectorDest[i] += vectorSrc[i];
    }
}

void addVectorFp64(const size_t dimension, double* vectorDest, double* vectorSrc)
{
    for(int i = 0; i < dimension; ++i)
    {
        vectorDest[i] += vectorSrc[i];
    }
}

//-------------------------------------------------------------

void expandMatrix(const size_t dimension, float* matrixFp32, double* matrixFp64)
{
    size_t countOfElements = dimension * dimension;

    for(int i = 0; i < countOfElements; ++i)
    {
        matrixFp64[i] = matrixFp32[i];
    }
}

//-------------------------------------------------------------

void expandVector(const size_t dimension, float* vectorFp32, double* vectorFp64)
{
    for(int i = 0; i < dimension; ++i)
    {
        vectorFp64[i] = vectorFp32[i];
    }
}

//-------------------------------------------------------------

int main()
{
    const size_t dim = 1000;    
    const size_t matrixElementsCount = dim * dim;
    const size_t matrixBytesCountFp32 = matrixElementsCount * sizeof(float);
    const size_t matrixBytesCountFp64 = matrixElementsCount * sizeof(double);

    float* matrixFp32 = (float*)malloc(matrixBytesCountFp32);
    double* matrixFp64 = (double*)malloc(matrixBytesCountFp64);

    float* luFp32 = (float*)malloc(matrixBytesCountFp32);
    double* luFp64 = (double*)malloc(matrixBytesCountFp64);

    float* vectorFp32 = (float*)malloc(dim * sizeof(float));
    double* vectorFp64 = (double*)malloc(dim * sizeof(double));

    float* knownX = (float*)malloc(dim * sizeof(float));

    double* residuals = (double*)malloc(dim * sizeof(double));

    int* permut = (int*)malloc(dim * sizeof(int));

    double* x = (double*)malloc(dim * sizeof(double));
    double* x2 = (double*)malloc(dim * sizeof(double));

    struct timeval before, after;

    gettimeofday(&before, NULL);
    srand(before.tv_usec);

    fillTestSystem(dim, knownX, matrixFp32, vectorFp32);

    memcpy(luFp32, matrixFp32, matrixBytesCountFp32);

    gettimeofday(&before, NULL);
    
    if(lupFactorization(dim, luFp32, permut))
    {   
        expandMatrix(dim, matrixFp32, matrixFp64);
        expandMatrix(dim, luFp32, luFp64);     
        expandVector(dim, vectorFp32, vectorFp64);

        solutionFp64(dim, luFp64, permut, vectorFp64, x);

        calcResidualsFp64(dim, x, matrixFp64, vectorFp64, residuals);

        printf("Err = %.7e\n", dotProductFp64(dim, residuals, residuals));   

        solutionFp64(dim, luFp64, permut, residuals, x2);

        addVectorFp64(dim, x, x2);

        calcResidualsFp64(dim, x, matrixFp64, vectorFp64, residuals);

        printf("Err = %.7e\n", dotProductFp64(dim, residuals, residuals));   

        gettimeofday(&after, NULL);

        printf("Done in %u ms.\n", timeDifference(&before, &after) / 1000);      
    }

    free(x2);
    free(x);
    free(permut);
    free(residuals);
    free(knownX);
    free(vectorFp64);
    free(vectorFp32);
    free(luFp64);
    free(luFp32);
    free(matrixFp64);
    free(matrixFp32);

    printf("Bye.\n");    
    
    return 0;
}

//-------------------------------------------------------------