#ifndef SOLVERFL_H
#define SOLVERFL_H

#include "ocl_playground.h"

//-------------------------------------------------------------

typedef void CL_CALLBACK fn_cl_notify
        (
            const char *errinfo, 
            const void *private_info, 
            size_t cb, 
            void *user_data
        );

//-------------------------------------------------------------

struct Solver
{
    ClPlayground* cl;

    cl_int errcode;

    cl_device_id dev;
    cl_context devco;
    cl_command_queue cmd;
    cl_program program;
    cl_kernel eliminatorKernel;

    cl_mem pp[2];   // ping-pong resources (augmented matrix)

    int ppInputIndex;
    int ppOutputIndex;

    size_t dimension;
    size_t aug_height;

    size_t bufferSize;

    size_t preferredWorkGroupSizeMultiple; 

    double* matrix;
    double* vector;

    int* history;   

    double* solution;

    double* residuals;

    double* iterativeSolution;

    double residualsSquaresSum;

    /////////////////////////////////////////
    Solver() : 
        devco(NULL), 
        cmd(NULL), 
        program(NULL), 
        eliminatorKernel(NULL), 
        preferredWorkGroupSizeMultiple(1),
        history(NULL),
        solution(NULL),
        residuals(NULL),
        residualsSquaresSum(0),
        iterativeSolution(NULL)
    {
        pp[0] = pp[1] = NULL;
    }

    /////////////////////////////////////////
    bool Init(size_t useDimension, ClPlayground* useCl, fn_cl_notify notify = NULL)
    {
        dimension = useDimension;
        aug_height = dimension + 1;

        bufferSize = dimension * aug_height * sizeof(double);

        history = (int*)malloc(dimension * sizeof(int));

        solution = (double*)malloc(dimension * sizeof(double));

        residuals = (double*)malloc(dimension * sizeof(double));

        iterativeSolution = (double*)malloc(dimension * sizeof(double));

            //

        cl = useCl;

        dev = cl->cpuDevices[0];

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

        program = cl->PrepareProgram(dev, devco, "#include \"kern_eliminator_double.cl\"");

        if(program == NULL)
        {
            return false;
        }

        eliminatorKernel = clCreateKernel(program, "eliminator", &errcode);

        if(eliminatorKernel == NULL)
        {
            return false;
        }

        errcode = clGetKernelWorkGroupInfo
        (
            eliminatorKernel, 
            dev, 
            CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
            sizeof(size_t), 
            &preferredWorkGroupSizeMultiple, 
            NULL
        );

        if(errcode != CL_SUCCESS)
        {
            preferredWorkGroupSizeMultiple = 1;
        }

        for(int i = 0; i < 2; ++i)
        {
            pp[i] = clCreateBuffer(devco, CL_MEM_READ_WRITE, bufferSize, NULL, &errcode);    
        }

        return (pp[0] != NULL) && (pp[1] != NULL);
    }

    /////////////////////////////////////////
    void useMatrix(double* argMatrix)
    {
        matrix = argMatrix;
    }

    /////////////////////////////////////////
    void useVector(double* argVector)
    {
        vector = argVector;    
    }

    /////////////////////////////////////////
    void Solve()
    {
            // map ping-pong buffers and copy transposed augmented data

        double* mappedBuffers[2];

        for(int i = 0; i < 2; ++i)
        {
            mappedBuffers[i] = (double*)clEnqueueMapBuffer(cmd, pp[i], CL_TRUE, CL_MAP_WRITE, 0, bufferSize, 0, NULL, NULL, &errcode);
        }
        
        int destOffset = 0;

        for(int row = 0; row < dimension; ++row)
        {
            for(int col = 0; col < dimension; ++col, ++destOffset)
            {                
                    // dest resource is row-major, but contains transposed initial matrix

                mappedBuffers[0][destOffset] = mappedBuffers[1][destOffset] = matrix[col * dimension + row];
            }
        }

            // augment with vector (last row of augmented resource)

        for(int col = 0; col < dimension; ++col, ++destOffset)
        {                
            mappedBuffers[0][destOffset] = mappedBuffers[1][destOffset] = vector[col];
        }

            // unmap buffers

        for(int i = 0; i < 2; ++i)
        {
            clEnqueueUnmapMemObject(cmd, pp[i], mappedBuffers[i], 0, NULL, NULL);
        }  

            // start main cycle
            
        ppInputIndex = 0;
        ppOutputIndex = 1;

        clSetKernelArg(eliminatorKernel, 0, sizeof(cl_uint), &dimension);

        size_t workOffset[2];
        size_t workSize[2];

        const size_t minGridSquare = 16 * 16;

        int left = 0;
        int top = 0;
        int leadingIndex = -1;

        for(; top < dimension; ++top)
        {               
            leadingIndex = findLeadingIndex(left, top);

            workOffset[0] = left;
            workOffset[1] = top;

            workSize[0] = dimension - left;
            workSize[1] = aug_height - top;

                // jump to CPU elimination when workSize becomes small 

            if(workSize[0] * workSize[1] < minGridSquare)
            {
                break;    
            }

                // setup args & execute kernel
            
            clSetKernelArg(eliminatorKernel, 1, sizeof(int), &left);
            clSetKernelArg(eliminatorKernel, 2, sizeof(int), &top);
            clSetKernelArg(eliminatorKernel, 3, sizeof(int), &leadingIndex);
            clSetKernelArg(eliminatorKernel, 4, sizeof(cl_mem), pp + ppInputIndex);
            clSetKernelArg(eliminatorKernel, 5, sizeof(cl_mem), pp + ppOutputIndex);
            
            clEnqueueNDRangeKernel(cmd, eliminatorKernel, 2, workOffset, workSize, NULL, 0, NULL, NULL);
            
            history[top] = left;

                // prepare next pass

            if(leadingIndex >= 0)
            {
                ++left;
            } 

            ppInputIndex = 1 - ppInputIndex;
            ppOutputIndex = 1 - ppOutputIndex;
        }

        continueElimination(left, top, leadingIndex);
    }

    /////////////////////////////////////////
    void CalcResiduals()
    {
        residualsSquaresSum = 0;

        for(int row = 0, offset = 0; row < dimension; ++row)
        {
            double s = 0;

            for(int col = 0; col < dimension; ++col, ++offset)
            {
                s += matrix[offset] * solution[col];
            }

            double e = vector[row] - s;

            residuals[row] = e;

            residualsSquaresSum += e * e;
        }
    }

    /////////////////////////////////////////
    void Iterate(int count)
    {        
        Solve();

        for(int j = 0; j < dimension; ++j)
        {
            iterativeSolution[j] = solution[j];
        }

        double* systemVector = vector;

        for(int i = 1; i < count; ++i)
        {            
            CalcResiduals();

            useVector(residuals);

            Solve();

            for(int j = 0; j < dimension; ++j)
            {
                iterativeSolution[j] = (solution[j] += iterativeSolution[j]);
            }                    

            useVector(systemVector);    
        }
    }

    /////////////////////////////////////////
    void Dispose()
    {
        free(iterativeSolution);
        free(residuals);
        free(solution);
        free(history);

        clReleaseMemObject(pp[0]);
        clReleaseMemObject(pp[1]);

        clReleaseKernel(eliminatorKernel);
        clReleaseProgram(program); 
        clReleaseCommandQueue(cmd);
        clReleaseContext(devco);
    }

private:

    /////////////////////////////////////////
    int findLeadingIndex(int left, int top)
    {
        int offset = (top * dimension + left) * sizeof(double);
        int count = dimension - left;
        int size = count * sizeof(double);

        double* mappedBuffer = (double*)clEnqueueMapBuffer(cmd, pp[ppInputIndex], CL_TRUE, CL_MAP_READ, offset, size, 0, NULL, NULL, &errcode);

        double maxElement = 0;
        int leadingIndex = -1;

        for(int i = 0; i < count; ++i)
        {
            double fv = fabs(mappedBuffer[i]);

            if(fv > maxElement)
            {
                maxElement = fv;
                leadingIndex = i;
            }
        }

        clEnqueueUnmapMemObject(cmd, pp[ppInputIndex], mappedBuffer, 0, NULL, NULL);

        return leadingIndex + ((leadingIndex < 0) ? 0 : left);
    }

    /////////////////////////////////////////
    void continueElimination(int left, int top, int leadingIndex)
    {
        double* dest = (double*)clEnqueueMapBuffer
        (
            cmd, 
            pp[ppInputIndex], 
            CL_TRUE, 
            CL_MAP_READ | CL_MAP_WRITE, 
            0, 
            bufferSize, 
            0, 
            NULL, 
            NULL, 
            &errcode
        );    

        double* src = (double*)clEnqueueMapBuffer
        (
            cmd, 
            pp[ppOutputIndex], 
            CL_TRUE, 
            CL_MAP_READ, 
            0, 
            bufferSize, 
            0, 
            NULL, 
            NULL, 
            &errcode
        );    

        // copy interleaved (ping-ponged) data:
        // top - 1, history[top - 1] - last target (now ppInput)
        // top - 2, history[top - 2] - prev.last target
        // top - 3, history[top - 3] - last target
        // ... 
        
        for(int i = top - 2; i >= 0; i -= 2)
        {            
            int colFrom = history[i];
            int colTo = history[i + 1];

            for(int row = i; row < aug_height; ++row)
            {
                for(int col = colFrom, offset = row * dimension + col; col < colTo; ++col, ++offset)
                {                 
                    dest[offset] = src[offset];    
                }
            }            
        }        

        clEnqueueUnmapMemObject(cmd, pp[ppOutputIndex], src, 0, NULL, NULL);

        // (left, top) points to 'unexplored' area, and leadingIndex is already found

        eliminate(left, top, leadingIndex, dest);
        
        backwardSubstitution(dest);

        clEnqueueUnmapMemObject(cmd, pp[ppInputIndex], dest, 0, NULL, NULL);
    }

    /////////////////////////////////////////
    void eliminate(int left, int top, int leadingIndex, double* dest)
    {
        for(; top < dimension;)
        {
            int topRowOffset = top * dimension;

            if(leadingIndex >= 0)
            {
                bool needSwap = (leadingIndex != left);

                int leadColPos = needSwap ? leadingIndex : left; 

                double leadDivisor = dest[topRowOffset + leadColPos];

                    // calc left column (now leading, after swap) 

                for(int row = top, offset = row * dimension; row < aug_height; ++row, offset += dimension)
                {                
                    double leadValue = dest[offset + leadColPos];

                    int offsetLeft = offset + left;

                    if(needSwap)
                    {
                        dest[offset + leadingIndex] = dest[offsetLeft];
                    }

                    dest[offsetLeft] = leadValue / leadDivisor;
                }

                    // calc other columns, subtracting 

                for(int col = left + 1; col < dimension; ++col)
                {
                    double divisor = dest[topRowOffset + col];

                    if(fabs(divisor) > 0)
                    {
                        for(int row = top, 
                                rowOffset = row * dimension, 
                                leftOffset = rowOffset + left, 
                                offset = rowOffset + col; 
                                
                                row < aug_height; 
                                
                                ++row, leftOffset += dimension, offset += dimension)
                        {
                            dest[offset] = dest[leftOffset] - dest[offset] / divisor;    
                        }
                    } 
                }    

                ++left;
            }

            ++top;

                // find next leading index (column with max top element)

            leadingIndex = -1;

            double maxElement = 0;
            
            for(int col = left, offset = top * dimension + left; col < dimension; ++col, ++offset)
            {
                double fv = fabs(dest[offset]);

                if(fv > maxElement)
                {
                    maxElement = fv;
                    leadingIndex = col;
                }
            }            

        } // end main cycle    

    } // end eliminate

    /////////////////////////////////////////
    void backwardSubstitution(double* dest)
    {
        int lastRowOffset = dimension * dimension;

        for(int i = dimension - 1; i >= 0; --i)
        {
            double s = 0;

            for(int j = i + 1; j < dimension; ++j)
            {
                s += dest[j * dimension + i] * solution[j];    
            }

            solution[i] = dest[lastRowOffset + i] - s;
        }
    }    
};

//-------------------------------------------------------------

#endif
