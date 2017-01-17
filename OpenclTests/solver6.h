#ifndef SOLVERFL_H
#define SOLVERFL_H

#include <string.h>

#include "ocl_playground.h"

//-------------------------------------------------------------

struct Solver
{
    ClPlayground* cl;

    cl_int errcode;

    cl_device_id dev;
    cl_context devco;
    cl_command_queue cmd;
    cl_program program;

    cl_kernel colSwapKernel;    
    cl_kernel rowDivideKernel;
    cl_kernel eliminatorKernel;

    cl_mem workspace;   // float (fp32) resource, for speed

    size_t dimension;

    size_t fp32VectorSize;
    size_t fp64VectorSize;

    size_t fp32ResourceSize;
    size_t fp64ResourceSize;

        // double (fp64) temporary data
    double* matrixFp64;
    double* vectorFp64;

        // permuatation vector, its element i (== p) shows that row p should stand instead of row i
    int* permutation;   

        // workspace expanded to fp64
    double* luFp64;     

        // double (fp64) post-factorization staff, vital
    double* solution;
    double* iterativeSolution;

    double* residuals;

    /////////////////////////////////////////
    Solver() : 
        devco(NULL), 
        cmd(NULL), 
        program(NULL), 
        colSwapKernel(NULL),
        rowDivideKernel(NULL),
        eliminatorKernel(NULL), 
        workspace(NULL),
        matrixFp64(NULL),
        vectorFp64(NULL),
        permutation(NULL),
        luFp64(NULL),
        solution(NULL),
        iterativeSolution(NULL),
        residuals(NULL)
    {}

    /////////////////////////////////////////
    void Dispose()
    {        
        free(residuals);
        free(iterativeSolution);
        free(solution);
        free(luFp64);
        free(permutation);
        free(vectorFp64);
        free(matrixFp64);
        
        clReleaseMemObject(workspace);

        clReleaseKernel(eliminatorKernel);
        clReleaseKernel(rowDivideKernel);
        clReleaseKernel(colSwapKernel);

        clReleaseProgram(program); 

        clReleaseCommandQueue(cmd);
        clReleaseContext(devco);
    }

    /////////////////////////////////////////
    bool Init(size_t useDimension, ClPlayground* useCl, fn_cl_notify notify = NULL)
    {
        dimension = useDimension;        

        fp32VectorSize = dimension * sizeof(float);
        fp64VectorSize = dimension * sizeof(double);

        fp32ResourceSize = fp32VectorSize * dimension;
        fp64ResourceSize = fp64VectorSize * dimension;

        matrixFp64 = (double*)malloc(fp64ResourceSize);
        vectorFp64 = (double*)malloc(fp64VectorSize);

        permutation = (int*)malloc((dimension - 1) * sizeof(int));

        luFp64 = (double*)malloc(fp64ResourceSize);

        solution = (double*)malloc(fp64VectorSize);
        iterativeSolution = (double*)malloc(fp64VectorSize);

        residuals = (double*)malloc(fp64VectorSize);

            // OpenCL stuff

        cl = useCl;

        dev = cl->gpuDevices[0];

        devco = clCreateContext(NULL, 1, &dev, notify, NULL, &errcode);

        if(devco == NULL)
        {
            return false;
        }
        
        cmd = clCreateCommandQueue(devco, dev, 0, &errcode);

        if(cmd == NULL)
        {
            return false;
        }

        program = cl->PrepareProgram(dev, devco, "#include \"kern_eliminator6.cl\"");

        if(program == NULL)
        {
            return false;
        }

        colSwapKernel = clCreateKernel(program, "col_swap", &errcode);
        rowDivideKernel = clCreateKernel(program, "row_divide", &errcode);
        eliminatorKernel = clCreateKernel(program, "eliminator", &errcode);

        if((colSwapKernel == NULL) || (rowDivideKernel == NULL) || (eliminatorKernel == NULL))
        {
            return false;
        }

        workspace = clCreateBuffer(devco, CL_MEM_READ_WRITE, fp32ResourceSize, NULL, &errcode);    

        return (workspace != NULL);
    }

    /////////////////////////////////////////
    void useMatrix(float* argMatrix)
    {
            // map workspace

        float* mappedWorkspace = (float*)clEnqueueMapBuffer
        (
            cmd, 
            workspace, 
            CL_TRUE, 
            CL_MAP_WRITE, 
            0, 
            fp32ResourceSize, 
            0, 
            NULL, 
            NULL, 
            &errcode
        );

            // make fp64 replica of argMatrix
            // and push transposed argMatrix to workspace
        
        for(size_t row = 0, index = 0; row < dimension; ++row)
        {
            for(size_t col = 0, transpIndex = row; col < dimension; ++col, ++index, transpIndex += dimension)
            {                
                float fv = argMatrix[index];

                mappedWorkspace[transpIndex] = fv;
                matrixFp64[index] = (double)fv;
            }
        }

            // unmap workspace from host memory

        clEnqueueUnmapMemObject(cmd, workspace, mappedWorkspace, 0, NULL, NULL);
    }

    /////////////////////////////////////////
    void useVector(float* argVector)
    {
        for(size_t i = 0; i < dimension; ++i)
        {
            vectorFp64[i] = argVector[i];
        }
    }

    /////////////////////////////////////////
    bool Solve()
    {
        //debprintMatrix(matrixFp64);

        printf("\n");

            // specify problem dimension to kernels

        clSetKernelArg(colSwapKernel, 0, sizeof(cl_uint), &dimension);
        clSetKernelArg(eliminatorKernel, 0, sizeof(cl_uint), &dimension);

            // specify global workspace to kernels

        clSetKernelArg(colSwapKernel, 3, sizeof(cl_mem), &workspace);
        clSetKernelArg(rowDivideKernel, 2, sizeof(cl_mem), &workspace);        
        clSetKernelArg(eliminatorKernel, 2, sizeof(cl_mem), &workspace);

        size_t workOffset[2];
        size_t workSize[2];

        unsigned int ldShift = dimension + 1;

        workSize[0] = dimension - 1;
        workSize[1] = workSize[0];

        workOffset[0] = 1;
        workOffset[1] = 1;

        float* mappedBuffer;

        for(unsigned int step = 0, mapOffset = 0, mapSize = dimension, ld = 1; 
                step < dimension; 
                    ++step, mapOffset += ldShift, --mapSize, ld += ldShift, --(workSize[0]), --(workSize[1]), ++(workOffset[0]), ++(workOffset[1]))
        {         
                // find pivot

            int pivotColIndex = -1;
            float pivotElement;
            float maxAbsValue = 0;

            mappedBuffer = (float*)clEnqueueMapBuffer
                                    (
                                        cmd, 
                                        workspace, 
                                        CL_TRUE, 
                                        CL_MAP_READ, 
                                        mapOffset * sizeof(float), 
                                        mapSize * sizeof(float), 
                                        0, 
                                        NULL, 
                                        NULL, 
                                        &errcode
                                    );

            for(unsigned int i = 0; i < mapSize; ++i)
            {
                float fv = mappedBuffer[i];
                float fav = fabs(fv);

                if(fav > maxAbsValue)
                {
                    pivotElement = fv;
                    maxAbsValue = fav;
                    pivotColIndex = i + step;
                }
            }

            clEnqueueUnmapMemObject(cmd, workspace, mappedBuffer, 0, NULL, NULL);

                /*
                printf("step: %u pivot: %d\n", step, pivotColIndex);
                debprintMatrixTmp();
                printf("\n");*/

            if(pivotColIndex < 0)
            {
                return false;
            }
                //

            if(workSize[0] > 0)
            {
                permutation[step] = pivotColIndex;

                if(pivotColIndex != step)
                {
                    // swap columns
                        
                    clSetKernelArg(colSwapKernel, 1, sizeof(cl_uint), &step);
                    clSetKernelArg(colSwapKernel, 2, sizeof(cl_uint), &pivotColIndex);                                

                    clEnqueueNDRangeKernel(cmd, colSwapKernel, 1, NULL, &dimension, NULL, 0, NULL, NULL);
                }

                    /*
                    printf("step: %u after swap\n", step);
                    debprintMatrixTmp();
                    printf("\n");*/

                    // divide row by pivotElement

                clSetKernelArg(rowDivideKernel, 0, sizeof(float), &pivotElement);
                clSetKernelArg(rowDivideKernel, 1, sizeof(cl_uint), &ld);

                clEnqueueNDRangeKernel(cmd, rowDivideKernel, 1, NULL, workSize, NULL, 0, NULL, NULL);

                    // eliminate

                clSetKernelArg(eliminatorKernel, 1, sizeof(cl_uint), &step);

                clEnqueueNDRangeKernel(cmd, eliminatorKernel, 2, workOffset, workSize, NULL, 0, NULL, NULL);
            }
        }

            // map workspace and expand (transposed) LU results 

        mappedBuffer = (float*)clEnqueueMapBuffer(cmd, workspace, CL_TRUE, CL_MAP_READ, 0, fp32ResourceSize, 0, NULL, NULL, &errcode);

        for(size_t row = 0, index = 0; row < dimension; ++row)
        {
            for(size_t col = 0, transpIndex = row; col < dimension; ++col, ++index, transpIndex += dimension)
            {                
                luFp64[transpIndex] = (double)(mappedBuffer[index]);
            }
        }
        
        clEnqueueUnmapMemObject(cmd, workspace, mappedBuffer, 0, NULL, NULL);

        //debprintMatrix(luFp64);

        useLupToSolve(solution, vectorFp64);

        return true;
    }

    /////////////////////////////////////////
    double CalcResiduals()
    {
        double residualsSquaresSum = 0;

        for(size_t row = 0, index = 0; row < dimension; ++row)
        {
            double s = 0;

            for(size_t col = 0; col < dimension; ++col, ++index)
            {
                s += matrixFp64[index] * solution[col];
            }

            double e = vectorFp64[row] - s;

            residuals[row] = e;

            residualsSquaresSum += e * e;
        }

        return residualsSquaresSum;
    }

    /////////////////////////////////////////
    bool Iterate(size_t count)
    {
        if(!Solve())
        {
            return false;
        }

        for(size_t i = 0; i < count; ++i)
        {
            CalcResiduals();

            useLupToSolve(iterativeSolution, residuals);

            for(size_t col = 0; col < dimension; ++col)
            {
                solution[col] += iterativeSolution[col];
            }
        }

        return true;    
    }

private:

    /////////////////////////////////////////
    void debprintMatrix(double* m)
    {
        for(size_t row = 0, index = 0; row < dimension; ++row)
        {
            for(size_t col = 0; col < dimension; ++col, ++index)
            {
                printf("%f ", m[index]);
            }

            printf("\n");
        }
    }

    /////////////////////////////////////////
    void debprintMatrixTmp()
    {
        float* m = (float*)clEnqueueMapBuffer
                                (
                                    cmd, 
                                    workspace, 
                                    CL_TRUE, 
                                    CL_MAP_READ, 
                                    0, 
                                    fp32ResourceSize, 
                                    0, 
                                    NULL, 
                                    NULL, 
                                    &errcode
                                );
        
        for(size_t row = 0; row < dimension; ++row)
        {
            for(size_t col = 0; col < dimension; ++col)
            {
                printf("%f ", m[col * dimension + row]);
            }

            printf("\n");
        }

        clEnqueueUnmapMemObject(cmd, workspace, m, 0, NULL, NULL);
    }

    /////////////////////////////////////////
    void useLupToSolve(double* x, double* b)
    {
        size_t lastIndexValue = dimension - 1;

        memcpy(x, b, fp64VectorSize);

            // permute right-hand part

        for(int row = 0; row < lastIndexValue; ++row)
        {
            int permutIndex = permutation[row];

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
                s += luFp64[index + col] * x[col];
            }

            x[row] -= s;
        }

            // Ux = y

        for(int row = lastIndexValue, index = row * dimension; row >= 0; --row, index -= dimension)
        {
            double s = 0;

            double de = luFp64[index + row];

            for(int col = row + 1; col < dimension; ++col)
            {
                s += luFp64[index + col] * x[col];
            }

            x[row] -= s;
            x[row] /= de;        
        }
    }
};

//-------------------------------------------------------------

#endif
