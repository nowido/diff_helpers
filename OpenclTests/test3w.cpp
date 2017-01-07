#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <windows.h>

#include "ocl_playground.h"

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

typedef void (CL_CALLBACK *fn_cl_notify)(const char *errinfo, const void *private_info, size_t cb, void *user_data);

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

    float* matrix;
    float* vector;

    /////////////////////////////////////////
    Solver() : devco(NULL), cmd(NULL), program(NULL), eliminatorKernel(NULL)
    {
        pp[0] = pp[1] = NULL;
    }

    /////////////////////////////////////////
    bool Init(size_t useDimension, ClPlayground* useCl, fn_cl_notify notify = NULL)
    {
        dimension = useDimension;
        aug_height = dimension + 1;

        bufferSize = dimension * aug_height * sizeof(float);

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

        program = cl->PrepareProgram(dev, devco, "#include \"kern_eliminator.cl\"");

        if(program == NULL)
        {
            return false;
        }

        eliminatorKernel = clCreateKernel(program, "eliminator", &errcode);

        if(eliminatorKernel == NULL)
        {
            return false;
        }

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

        clSetKernelArg(eliminatorKernel, 0, sizeof(cl_uint), &dimension);

        size_t workOffset[2];
        size_t workSize[2];

        for(int left = 0, top = 0; top < dimension;)
        {               
            int leadingIndex = findLeadingIndex(left, top);

                // setup args & execute kernel
            
            clSetKernelArg(eliminatorKernel, 1, sizeof(int), &left);
            clSetKernelArg(eliminatorKernel, 2, sizeof(int), &top);
            clSetKernelArg(eliminatorKernel, 3, sizeof(int), &leadingIndex);
            clSetKernelArg(eliminatorKernel, 4, sizeof(cl_mem), pp + ppInputIndex);
            clSetKernelArg(eliminatorKernel, 5, sizeof(cl_mem), pp + ppOutputIndex);
            
            workOffset[0] = left;
            workOffset[1] = top;

            workSize[0] = dimension - left;
            workSize[1] = aug_height - top;

            clEnqueueNDRangeKernel(cmd, eliminatorKernel, 2, workOffset, workSize, NULL, 0, NULL, NULL);

            if(leadingIndex >= 0)
            {
                ++left;
            } 

            ++top;

            ppInputIndex = 1 - ppInputIndex;
            ppOutputIndex = 1 - ppOutputIndex;
        }
    }

    /////////////////////////////////////////
    void Dispose()
    {
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
};

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

            float kx = knownX[col];

            s += mv * kx;
        }

        vector[row] = s;
    }
}

//-------------------------------------------------------------

int main()
{
    const size_t dim = 1000;

    float* matrix = (float*)malloc(dim * dim * sizeof(float));
    float* vector = (float*)malloc(dim * sizeof(float));
    float* knownX = (float*)malloc(dim * sizeof(float));

    fillTestSystem(dim, knownX, matrix, vector);

        //

    FILETIME before, after;

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
    GetSystemTimeAsFileTime(&after);

    printf("Execution time: %u ms.\n", timeDifference(&before, &after) / 1000);

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
