#ifndef SOLVERFL_H
#define SOLVERFL_H

#include <string.h>

#include <xmmintrin.h>

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
    cl_kernel eliminatorKernel;

    cl_event r1mapped;
    cl_event r2mapped;
    cl_event r1gpuMainDone;
    cl_event r2gpuMainDone;

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
    double* matrix2_Fp64;

    double* vectorFp64;

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
        eliminatorKernel(NULL), 
        r1mapped(NULL),
        r2mapped(NULL),
        r1gpuMainDone(NULL),
        r2gpuMainDone(NULL),
        workspace1(NULL),
        workspace2(NULL),
        matrix1_Fp64(NULL),
        matrix2_Fp64(NULL),
        vectorFp64(NULL),
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
        free(vectorFp64); 
        free(matrix1_Fp64);
        free(matrix2_Fp64);
        
        clReleaseMemObject(workspace1);
        clReleaseMemObject(workspace2);

        clReleaseEvent(r1mapped);
        clReleaseEvent(r2mapped);
        clReleaseEvent(r1gpuMainDone);
        clReleaseEvent(r2gpuMainDone);

        clReleaseKernel(eliminatorKernel);        
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

        vectorFp64 = (double*)malloc(fp64VectorSize);                

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
        eliminatorKernel = clCreateKernel(program, "eliminator", &errcode);

        if((colSwapKernel == NULL) || (eliminatorKernel == NULL))
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

            // do the same with the second replica, but store columns in reverse mode

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
        for(size_t row = 0; row < dimension; ++row)
        {                        
            vectorFp64[row] = (double)argVector[row];            
        }
    }

    /////////////////////////////////////////
    bool Solve()
    {
        // to do 

        alignas(16) float a[4] = {1, 2, 3, 4};
        alignas(16) float b[4] = {20, -3, 14, 15};
        alignas(16) float c[4];

        __m128* a_simd = reinterpret_cast<__m128*>(a);
        __m128* b_simd = reinterpret_cast<__m128*>(b);
        __m128* c_simd = reinterpret_cast<__m128*>(c);

        _mm_store_ps(c, _mm_add_ps(*a_simd, *b_simd));

        groupSizeX = 16;
        groupSizeY = 12;

            // specify problem dimension to kernels

        clSetKernelArg(colSwapKernel, 0, sizeof(cl_uint), &dimension);
        clSetKernelArg(eliminatorKernel, 0, sizeof(cl_uint), &dimension);

            // specify local blocks (allocation sizes)

        clSetKernelArg(eliminatorKernel, 3, groupSizeX * sizeof(float), NULL);    
        clSetKernelArg(eliminatorKernel, 4, groupSizeY * sizeof(float), NULL);    

            //

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

        float* mappedBuffer1;
        float* mappedBuffer2;

        bool inconsistent = false;

        int pivotOffset;
        int pivotColIndex;
        float pivotElement;
        float maxAbsValue;

        for(unsigned int step = 0, mapOffset = 0, mapSize = dimension, ld = 1; 
                step < dimension; 
                    ++step, mapOffset += ldShift, --mapSize, ld += ldShift, --(workSize[0]), --(workSize[1]), ++(workOffset[0]), ++(workOffset[1]))
        {         
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
            
                // enqueue map r1 non-blocking, depending on r1gpuMainDone, ret r1mapped

            mappedBuffer1 = (float*)clEnqueueMapBuffer
                                    (
                                        cmd, 
                                        workspace1, 
                                        step ? CL_FALSE : CL_TRUE, 
                                        CL_MAP_READ | CL_MAP_WRITE, 
                                        mapOffset * sizeof(float), 
                                        mapSize * sizeof(float), 
                                        step ? 1 : 0, 
                                        step ? &r1gpuMainDone : NULL, 
                                        step ? &r1mapped : NULL, 
                                        &errcode
                                    );

                // enqueue map r2 non-blocking, depending on r2gpuMainDone, ret r2mapped

            mappedBuffer2 = (float*)clEnqueueMapBuffer
                                    (
                                        cmd, 
                                        workspace2, 
                                        step ? CL_FALSE : CL_TRUE, 
                                        CL_MAP_READ | CL_MAP_WRITE, 
                                        mapOffset * sizeof(float), 
                                        mapSize * sizeof(float), 
                                        step ? 1 : 0, 
                                        step ? &r2gpuMainDone : NULL, 
                                        step ? &r2mapped : NULL, 
                                        &errcode
                                    );
                
                // wait r1mapped

            clWaitForEvents(1, &r1mapped);

                // work with r1

                // find pivot

            
            pivotColIndex = -1;            
            maxAbsValue = 0;

            for(unsigned int i = 0; i < mapSize; ++i)
            {
                float fv = mappedBuffer1[i];
                //float fav = fabs(fv);
                float fav = fv;

                    // clear sign bit to get absolute value
                ((char*)&fav)[3] &= '\x7F';                

                if(fav > maxAbsValue)
                {
                    pivotElement = fv;
                    maxAbsValue = fav;
                    pivotOffset = i;
                    pivotColIndex = i + step;
                }
            }

            if(pivotColIndex < 0)
            {
                inconsistent = true;                
            }
            else
            {
                    // divide row by pivotElement

                for(unsigned int i = 0; i < mapSize; ++i)
                {
                    if(pivotOffset != (int)i)
                    {
                        mappedBuffer1[i] /= pivotElement;    
                    }
                }
            }
            
                // unmap r1

            clEnqueueUnmapMemObject(cmd, workspace1, mappedBuffer1, 0, NULL, NULL);
                
                //

            if(!inconsistent && (workSize[0] > 0))
            {
                    // enqueue kernels on r1

                    // specify global workspace to kernels

                clSetKernelArg(colSwapKernel, 3, sizeof(cl_mem), &workspace1);
                clSetKernelArg(eliminatorKernel, 2, sizeof(cl_mem), &workspace1);

                permutation1[step] = pivotColIndex;

                if(pivotColIndex != step)
                {
                    // swap columns
                        
                    clSetKernelArg(colSwapKernel, 1, sizeof(cl_uint), &step);
                    clSetKernelArg(colSwapKernel, 2, sizeof(cl_uint), &pivotColIndex);                                

                    clEnqueueNDRangeKernel(cmd, colSwapKernel, 1, NULL, &dimension, NULL, 0, NULL, NULL);
                }

                    // eliminate

                clSetKernelArg(eliminatorKernel, 1, sizeof(cl_uint), &step);
                                
                clEnqueueNDRangeKernel(cmd, eliminatorKernel, 2, workOffset, actualGlobalSize, localSize, 0, NULL, &r1gpuMainDone);

            } // end enqueue kernels on r1

                // wait r2mapped

            clWaitForEvents(1, &r2mapped);

                // work with r2

                // find pivot

            pivotColIndex = -1;            
            maxAbsValue = 0;

            for(unsigned int i = 0; i < mapSize; ++i)
            {
                float fv = mappedBuffer2[i];
                //float fav = fabs(fv);
                float fav = fv;

                    // clear sign bit to get absolute value
                ((char*)&fav)[3] &= '\x7F';                

                if(fav > maxAbsValue)
                {
                    pivotElement = fv;
                    maxAbsValue = fav;
                    pivotOffset = i;
                    pivotColIndex = i + step;
                }
            }
                
            if(pivotColIndex < 0)
            {
                inconsistent = true;                
            }
            else
            {
                    // divide row by pivotElement

                for(unsigned int i = 0; i < mapSize; ++i)
                {
                    if(pivotOffset != (int)i)
                    {
                        mappedBuffer2[i] /= pivotElement;    
                    }
                }
            }
                
                // unmap r2

            clEnqueueUnmapMemObject(cmd, workspace2, mappedBuffer2, 0, NULL, NULL);    

                //

            if(!inconsistent && (workSize[0] > 0))
            {
                    // enqueue kernels on r2
                
                    // specify global workspace to kernels

                clSetKernelArg(colSwapKernel, 3, sizeof(cl_mem), &workspace2);                
                clSetKernelArg(eliminatorKernel, 2, sizeof(cl_mem), &workspace2);
                
                permutation2[step] = pivotColIndex;

                if(pivotColIndex != step)
                {
                    // swap columns
                        
                    clSetKernelArg(colSwapKernel, 1, sizeof(cl_uint), &step);
                    clSetKernelArg(colSwapKernel, 2, sizeof(cl_uint), &pivotColIndex);                                

                    clEnqueueNDRangeKernel(cmd, colSwapKernel, 1, NULL, &dimension, NULL, 0, NULL, NULL);
                }

                    // eliminate

                clSetKernelArg(eliminatorKernel, 1, sizeof(cl_uint), &step);

                clEnqueueNDRangeKernel(cmd, eliminatorKernel, 2, workOffset, actualGlobalSize, localSize, 0, NULL, &r2gpuMainDone);
            }

            if(inconsistent)
            {
                return false;
            }
        }

            // map workspace and expand (transposed) LU results 

        mappedBuffer1 = (float*)clEnqueueMapBuffer
                                (
                                    cmd, 
                                    workspace1, 
                                    CL_TRUE, 
                                    CL_MAP_READ, 
                                    0, 
                                    fp32ResourceSize, 
                                    0, 
                                    NULL, 
                                    NULL, 
                                    &errcode
                                );

        mappedBuffer2 = (float*)clEnqueueMapBuffer
                                (
                                    cmd, 
                                    workspace2, 
                                    CL_TRUE, 
                                    CL_MAP_READ, 
                                    0, 
                                    fp32ResourceSize, 
                                    0, 
                                    NULL, 
                                    NULL, 
                                    &errcode
                                );

        for(size_t row = 0, index = 0; row < dimension; ++row)
        {
            for(size_t col = 0, transpIndex = row; col < dimension; ++col, ++index, transpIndex += dimension)
            {                
                lu1_Fp64[transpIndex] = (double)(mappedBuffer1[index]);
                lu2_Fp64[transpIndex] = (double)(mappedBuffer2[index]);
            }
        }
        
        clEnqueueUnmapMemObject(cmd, workspace1, mappedBuffer1, 0, NULL, NULL);
        clEnqueueUnmapMemObject(cmd, workspace2, mappedBuffer2, 0, NULL, NULL);

        useLupToSolve(permutation1, lu1_Fp64, solution1, vectorFp64);
        useLupToSolve(permutation2, lu2_Fp64, solution2, vectorFp64);

        return true;
    }

    /////////////////////////////////////////
    double CalcResiduals(double* m, double* x, double* r)
    {
        double residualsSquaresSum = 0;

        for(size_t row = 0, index = 0; row < dimension; ++row)
        {
            double s = 0;

            for(size_t col = 0; col < dimension; ++col, ++index)
            {
                s += m[index] * x[col];
            }

            double e = vectorFp64[row] - s;

            r[row] = e;

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
            CalcResiduals(matrix1_Fp64, solution1, residuals1);
            CalcResiduals(matrix2_Fp64, solution2, residuals2);

            useLupToSolve(permutation1, lu1_Fp64, iterativeSolution1, residuals1);
            useLupToSolve(permutation2, lu2_Fp64, iterativeSolution2, residuals2);

            for(size_t col = 0; col < dimension; ++col)
            {
                solution1[col] += iterativeSolution1[col];
                solution2[col] += iterativeSolution2[col];
            }
        }

        return true;    
    }

    /////////////////////////////////////////
    void useLupToSolve(int* p, double* lu, double* x, double* b)
    {
        memcpy(x, b, fp64VectorSize);

            // permute right-hand part

        for(int row = 0; row < lastIndexValue; ++row)
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

        for(int row = lastIndexValue, index = row * dimension; row >= 0; --row, index -= dimension)
        {
            double s = 0;

            double de = lu[index + row];

            for(int col = row + 1; col < dimension; ++col)
            {
                s += lu[index + col] * x[col];
            }

            x[row] -= s;
            x[row] /= de;        
        }
    }
};

//-------------------------------------------------------------

#endif
