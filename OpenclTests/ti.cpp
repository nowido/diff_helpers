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

            s += mv * knownX[col];
        }

        vector[row] = s;
    }
}

//-------------------------------------------------------------

float dotProduct(const size_t dimension, float* x1, float* x2)
{
    float s = 0;

    for(int i = 0; i < dimension; ++i)
    {
        s += x1[i] * x2[i];
    }

    return s;
}

//-------------------------------------------------------------

void calcResiduals(const size_t dimension, float* x, float* matrix, float* vector, float* residuals)
{
    for(int row = 0; row < dimension; ++row)
    {
        residuals[row] = vector[row] - dotProduct(dimension, matrix + row * dimension, x);
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

void solution(const size_t dimension, float* lu, int* p, float* vector, float* x)
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

//-------------------------------------------------------------

void printMatrix(const size_t dimension, float* matrix)
{
    for(int row = 0, index = 0; row < dimension; ++row)
    {
        for(int col = 0; col < dimension; ++col, ++index)
        {
            printf("%f ", matrix[index]);
        }

        printf("\n");
    }
}

//-------------------------------------------------------------

void addVector(const size_t dimension, float* vectorDest, float* vectorSrc)
{
    for(int i = 0; i < dimension; ++i)
    {
        vectorDest[i] += vectorSrc[i];
    }
}

//-------------------------------------------------------------

void printVector(const size_t dimension, float* vector)
{
    for(int row = 0; row < dimension; ++row)
    {
        printf("%f ", vector[row]);
    }

    printf("\n");
}

//-------------------------------------------------------------

int main()
{
    const size_t dim = 1000;    
    const size_t matrixElementsCount = dim * dim;
    const size_t matrixBytesCount = matrixElementsCount * sizeof(float);

    float* matrix = (float*)malloc(matrixBytesCount);
    float* lu = (float*)malloc(matrixBytesCount);
    float* vector = (float*)malloc(dim * sizeof(float));
    float* knownX = (float*)malloc(dim * sizeof(float));
    float* residuals = (float*)malloc(dim * sizeof(float));
    int* permut = (int*)malloc(dim * sizeof(int));
    float* x = (float*)malloc(dim * sizeof(float));
    float* x2 = (float*)malloc(dim * sizeof(float));

    struct timeval before, after;

    gettimeofday(&before, NULL);
    srand(before.tv_usec);

    fillTestSystem(dim, knownX, matrix, vector);

    /*
    printMatrix(dim, matrix);

    printf("==\n");
    printVector(dim, vector);

    printf("kx==\n");
    printVector(dim, knownX);
    */
    memcpy(lu, matrix, matrixBytesCount);

    gettimeofday(&before, NULL);
    
    if(lupFactorization(dim, lu, permut))
    {        
        solution(dim, lu, permut, vector, x);

        //printMatrix(dim, lu);

        calcResiduals(dim, x, matrix, vector, residuals);

        printf("Err = %f\n", dotProduct(dim, residuals, residuals));   

        solution(dim, lu, permut, residuals, x2);

        addVector(dim, x, x2);

        calcResiduals(dim, x, matrix, vector, residuals);

        printf("Err = %f\n", dotProduct(dim, residuals, residuals));   

        gettimeofday(&after, NULL);

        printf("Done in %u ms.\n", timeDifference(&before, &after) / 1000);      
    }

    free(x2);
    free(x);
    free(permut);
    free(residuals);
    free(knownX);
    free(vector);
    free(lu);
    free(matrix);

    printf("Bye.\n");    
    
    return 0;
}

//-------------------------------------------------------------