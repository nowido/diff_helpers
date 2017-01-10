#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <windows.h>

#include "ocl_playground.h"
#include "solverdbl.h"

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

double getRandom(double amplitude)
{
    return floor(amplitude * ((double)rand() / (double)RAND_MAX));
}

//-------------------------------------------------------------

void fillTestSystem(const size_t dimension, double* knownX, double* matrix, double* vector)
{
    const double amplitude = 10;

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

            double kx = knownX[col];

            s += mv * kx;
        }

        vector[row] = s;
    }
}

//-------------------------------------------------------------

int main()
{
    const size_t dim = 1000;

    double* matrix = (double*)malloc(dim * dim * sizeof(double));
    double* vector = (double*)malloc(dim * sizeof(double));
    double* knownX = (double*)malloc(dim * sizeof(double));

        //

    FILETIME before, after;

    GetSystemTimeAsFileTime(&before);
    srand(before.dwLowDateTime);

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

    GetSystemTimeAsFileTime(&before);
    slv.Solve();
    //slv.Iterate(3);
    GetSystemTimeAsFileTime(&after);

    printf("Execution time: %u ms.\n", timeDifference(&before, &after) / 1000);

    slv.CalcResiduals();

    printf("Residuals squares sum: %lf\n", slv.residualsSquaresSum);

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