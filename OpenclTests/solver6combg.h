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

        // float (fp32) resources, for speed
    cl_mem workspace1;   
    cl_mem workspace2;   

    cl_uint groupSizeX;
    cl_uint groupSizeY;

    size_t dimension;
    size_t lastIndexValue;

    size_t fp32VectorSize;
    size_t fp64VectorSize;

    size_t fp32ResourceSize;
    size_t fp64ResourceSize;

        // double (fp64) temporary data
    double* matrix1_Fp64;
    double* vector1_Fp64;

        // permuatation vector, its element i (== p) shows that row p should stand instead of row i
    int* permutation1;   
    int* permutation2;   

        // workspace expanded to fp64
    double* lu1_Fp64;
    double* lu2_Fp64;     

        // double (fp64) post-factorization staff, vital
    double* solution1;
    double* solution2;

    double* iterativeSolution1;
    double* iterativeSolution2;

    double* residuals1;
    double* residuals2;

    /////////////////////////////////////////
    Solver() : 
        devco(NULL), 
        cmd(NULL), 
        program(NULL), 
        colSwapKernel(NULL),
        rowDivideKernel(NULL),
        eliminatorKernel(NULL), 
        workspace1(NULL),
        workspace2(NULL),
        matrix1_Fp64(NULL),
        vector1_Fp64(NULL),
        matrix2_Fp64(NULL),
        vector2_Fp64(NULL),        
        permutation1(NULL),
        permutation2(NULL),
        lu1_Fp64(NULL),
        lu2_Fp64(NULL),
        solution1(NULL),
        solution2(NULL),
        iterativeSolution1(NULL),
        iterativeSolution2(NULL),
        residuals1(NULL),
        residuals2(NULL)
    {}

    /////////////////////////////////////////
    void Dispose()
    {        
        free(residuals1);
        free(residuals2);        
        free(iterativeSolution1);
        free(iterativeSolution2);
        free(solution1);
        free(solution2);
        free(lu1_Fp64);
        free(lu2_Fp64);
        free(permutation1);
        free(permutation2);
        free(vector1_Fp64);
        free(vector2_Fp64);
        free(matrix1_Fp64);
        free(matrix2_Fp64);
        
        clReleaseMemObject(workspace1);
        clReleaseMemObject(workspace2);

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
            // resource sizes

        dimension = useDimension;      
        lastIndexValue = dimension - 1;  

        fp32VectorSize = dimension * sizeof(float);
        fp64VectorSize = dimension * sizeof(double);

        fp32ResourceSize = fp32VectorSize * dimension;
        fp64ResourceSize = fp64VectorSize * dimension;

            // memory allocation

        matrix1_Fp64 = (double*)malloc(fp64ResourceSize);
        matrix2_Fp64 = (double*)malloc(fp64ResourceSize);

        vector1_Fp64 = (double*)malloc(fp64VectorSize);        
        vector2_Fp64 = (double*)malloc(fp64VectorSize);

        permutation1 = (int*)malloc(lastIndexValue * sizeof(int));
        permutation2 = (int*)malloc(lastIndexValue * sizeof(int));

        lu1_Fp64 = (double*)malloc(fp64ResourceSize);
        lu2_Fp64 = (double*)malloc(fp64ResourceSize);

        solution1 = (double*)malloc(fp64VectorSize);
        solution2 = (double*)malloc(fp64VectorSize);

        iterativeSolution1 = (double*)malloc(fp64VectorSize);
        iterativeSolution2 = (double*)malloc(fp64VectorSize);

        residuals1 = (double*)malloc(fp64VectorSize);
        residuals2 = (double*)malloc(fp64VectorSize);

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

        program = cl->PrepareProgram(dev, devco, "#include \"kern_eliminator6g.cl\"");

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

        workspace1 = clCreateBuffer(devco, CL_MEM_READ_WRITE, fp32ResourceSize, NULL, &errcode);    
        workspace2 = clCreateBuffer(devco, CL_MEM_READ_WRITE, fp32ResourceSize, NULL, &errcode);    

        return (workspace1 != NULL) && (workspace2 != NULL);
    }

    /////////////////////////////////////////
    void useMatrix(float* argMatrix)
    {
            // map workspace

        float* mappedWorkspace1 = (float*)clEnqueueMapBuffer
        (
            cmd, 
            workspace1, 
            CL_TRUE, 
            CL_MAP_WRITE, 
            0, 
            fp32ResourceSize, 
            0, 
            NULL, 
            NULL, 
            &errcode
        );

        float* mappedWorkspace2 = (float*)clEnqueueMapBuffer
        (
            cmd, 
            workspace2, 
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

                mappedWorkspace1[transpIndex] = fv;

                matrix1_Fp64[index] = (double)fv;
            }
        }

            // do the same with the second replica, but reverse columns

        for(size_t row = 0, rowOffset = 0; row < dimension; ++row, rowOffset += dimension)
        {
            for(size_t col = 0, colRev = lastIndexValue, transpIndex = row; col < dimension; ++col, --colRev, transpIndex += dimension)
            {   
                float fv = argMatrix[rowOffset + colRev];

                mappedWorkspace2[transpIndex] = fv;

                matrix2_Fp64[rowOffset + col] = (double)fv;
            }
        }

            // unmap workspace from host memory

        clEnqueueUnmapMemObject(cmd, workspace1, mappedWorkspace1, 0, NULL, NULL);
        clEnqueueUnmapMemObject(cmd, workspace2, mappedWorkspace2, 0, NULL, NULL);
    }

    /////////////////////////////////////////
    void useVector(float* argVector)
    {
        for(size_t row = 0, rowRev = lastIndexValue; row < dimension; ++row, --rowRev)
        {
            vector1_Fp64[row] = argVector[row];
            
        }
    }

    /////////////////////////////////////////
    bool Solve()
    {
        //debprintMatrix(matrixFp64);
        //printf("\n");

        groupSizeX = 16;
        groupSizeY = 12;

            // specify problem dimension to kernels

        clSetKernelArg(colSwapKernel, 0, sizeof(cl_uint), &dimension);
        clSetKernelArg(eliminatorKernel, 0, sizeof(cl_uint), &dimension);

            // specify global workspace to kernels

        clSetKernelArg(colSwapKernel, 3, sizeof(cl_mem), &workspace);
        clSetKernelArg(rowDivideKernel, 2, sizeof(cl_mem), &workspace);        
        clSetKernelArg(eliminatorKernel, 2, sizeof(cl_mem), &workspace);

            // specify local blocks (allocation sizes)
        clSetKernelArg(eliminatorKernel, 3, groupSizeX * sizeof(float), NULL);    
        clSetKernelArg(eliminatorKernel, 4, groupSizeY * sizeof(float), NULL);    

        size_t workOffset[2];
        size_t workSize[2];
        size_t localSize[2];
        size_t actualGlobalSize[2];

        unsigned int ldShift = dimension + 1;

        workSize[0] = dimension - 1;
        workSize[1] = workSize[0];

        workOffset[0] = 1;
        workOffset[1] = 1;

        localSize[0] = groupSizeX;
        localSize[1] = groupSizeY;

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
                //float fav = fabs(fv);
                float fav = fv;

                    // clear sign bit to get absolute value
                ((char*)&fav)[3] &= '\x7F';                

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

                // correct global size
                actualGlobalSize[0] = workSize[0];
                actualGlobalSize[1] = workSize[1];

                size_t rm = actualGlobalSize[0] % groupSizeX;

                if(rm)
                {
                    actualGlobalSize[0] += groupSizeX - rm;
                }
                
                rm = actualGlobalSize[1] % groupSizeY;

                if(rm)
                {
                    actualGlobalSize[1] += groupSizeY - rm;
                }
                                
                clEnqueueNDRangeKernel(cmd, eliminatorKernel, 2, workOffset, actualGlobalSize, localSize, 0, NULL, NULL);
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

public:

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
