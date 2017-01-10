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

    size_t workGroupSizeX;
    size_t workGroupSizeY;

    float* matrix;
    float* vector;

    int* history;   

    float* solution;

    float* residuals;

    float* iterativeSolution;

    float residualsSquaresSum;

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

        bufferSize = dimension * aug_height * sizeof(float);

        history = (int*)malloc(dimension * sizeof(int));

        solution = (float*)malloc(dimension * sizeof(float));

        residuals = (float*)malloc(dimension * sizeof(float));

        iterativeSolution = (float*)malloc(dimension * sizeof(float));

            //

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

        program = cl->PrepareProgram(dev, devco, "#include \"kern_eliminator_flgr.cl\"");

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

        workGroupSizeX = 256;    // to do: use preferredWorkGroupSizeMultiple
        workGroupSizeY = 1;

        for(int i = 0; i < 2; ++i)
        {
            pp[i] = clCreateBuffer(devco, CL_MEM_READ_WRITE, bufferSize, NULL, &errcode);    
        }

        return (pp[0] != NULL) && (pp[1] != NULL);
    }

    /////////////////////////////////////////
    void useMatrix(float* argMatrix)
    {
        matrix = argMatrix;
    }

    /////////////////////////////////////////
    void useVector(float* argVector)
    {
        vector = argVector;    
    }

    /////////////////////////////////////////
    void Solve()
    {
            // map ping-pong buffers and copy transposed augmented data

        float* mappedBuffers[2];

        for(int i = 0; i < 2; ++i)
        {
            mappedBuffers[i] = (float*)clEnqueueMapBuffer(cmd, pp[i], CL_TRUE, CL_MAP_WRITE, 0, bufferSize, 0, NULL, NULL, &errcode);
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

                // specify problem dimension to kernel

        clSetKernelArg(eliminatorKernel, 0, sizeof(cl_uint), &dimension);

                // allocate proper local arrays to kernel

        clSetKernelArg(eliminatorKernel, 4, workGroupSizeY * sizeof(float), NULL);
        clSetKernelArg(eliminatorKernel, 5, workGroupSizeX * sizeof(float), NULL);        
        clSetKernelArg(eliminatorKernel, 6, workGroupSizeX/4 * sizeof(int), NULL);        
        
        size_t workOffset[2];
        size_t workSize[2];

        size_t localWorkSize[2];

        localWorkSize[0] = workGroupSizeX;
        localWorkSize[1] = workGroupSizeY;

        const size_t minGridSquare = 64 * 64;

        int left = 0;
        int top = 0;
        int leadingIndex = -1;

        for(; top < dimension; ++top)
        {               
            leadingIndex = findLeadingIndex(left, top);

            workOffset[0] = left;
            workOffset[1] = top;

            size_t workRectWidth = dimension - left;
            size_t workRectHeight = aug_height - top;

                // jump to CPU elimination when workSize becomes small 

            if(workRectWidth * workRectHeight < minGridSquare)
            {
                break;    
            }

                // ensure global work size is evenly divisible by group size

            size_t remX = workRectWidth % workGroupSizeX;
            size_t remY = workRectHeight % workGroupSizeY;

            workSize[0] = workRectWidth + ((remX == 0) ? 0 : (workGroupSizeX - remX));
            workSize[1] = workRectHeight + ((remY == 0) ? 0 : (workGroupSizeY - remY));

                // setup args & execute kernel
            
                    // global 'constants'

            clSetKernelArg(eliminatorKernel, 1, sizeof(int), &left);
            clSetKernelArg(eliminatorKernel, 2, sizeof(int), &top);
            clSetKernelArg(eliminatorKernel, 3, sizeof(int), &leadingIndex);

                    // ping-ponged input and output resources

            clSetKernelArg(eliminatorKernel, 7, sizeof(cl_mem), pp + ppInputIndex);
            clSetKernelArg(eliminatorKernel, 8, sizeof(cl_mem), pp + ppOutputIndex);

            clEnqueueNDRangeKernel(cmd, eliminatorKernel, 2, workOffset, workSize, localWorkSize, 0, NULL, NULL);
            
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
            float s = 0;

            for(int col = 0; col < dimension; ++col, ++offset)
            {
                s += matrix[offset] * solution[col];
            }

            float e = vector[row] - s;

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

        float* systemVector = vector;

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
        int offset = (top * dimension + left) * sizeof(float);
        int count = dimension - left;
        int size = count * sizeof(float);

        float* mappedBuffer = (float*)clEnqueueMapBuffer(cmd, pp[ppInputIndex], CL_TRUE, CL_MAP_READ, offset, size, 0, NULL, NULL, &errcode);

        float maxElement = 0;
        int leadingIndex = -1;

        for(int i = 0; i < count; ++i)
        {
            float fv = fabs(mappedBuffer[i]);

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
        float* dest = (float*)clEnqueueMapBuffer
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

        float* src = (float*)clEnqueueMapBuffer
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
    void eliminate(int left, int top, int leadingIndex, float* dest)
    {
        for(; top < dimension;)
        {
            int topRowOffset = top * dimension;

            if(leadingIndex >= 0)
            {
                bool needSwap = (leadingIndex != left);

                int leadColPos = needSwap ? leadingIndex : left; 

                float leadDivisor = dest[topRowOffset + leadColPos];

                    // calc left column (now leading, after swap) 

                for(int row = top, offset = row * dimension; row < aug_height; ++row, offset += dimension)
                {                
                    float leadValue = dest[offset + leadColPos];

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
                    float divisor = dest[topRowOffset + col];

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

            float maxElement = 0;
            
            for(int col = left, offset = top * dimension + left; col < dimension; ++col, ++offset)
            {
                float fv = fabs(dest[offset]);

                if(fv > maxElement)
                {
                    maxElement = fv;
                    leadingIndex = col;
                }
            }            

        } // end main cycle    

    } // end eliminate

    /////////////////////////////////////////
    void backwardSubstitution(float* dest)
    {
        int lastRowOffset = dimension * dimension;

        for(int i = dimension - 1; i >= 0; --i)
        {
            float s = 0;

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
