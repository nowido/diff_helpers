#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <sys/time.h>
#include <unistd.h>

#include "ocl_playground.h"

//-------------------------------------------------------------

unsigned int timeDifference(struct timeval *before, struct timeval *after)
{
    return ((after->tv_sec * 1000000 + after->tv_usec) - (before->tv_sec * 1000000 + before->tv_usec));
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

void fillSrcDataRandom(float* buffer, size_t numElements)
{
    for(int i = 0; i < numElements; ++i)
    {
        buffer[i] = 1000.0f * ((float)rand() / (float)RAND_MAX);
    }
}

//-------------------------------------------------------------

float calcErr(float* src, float* processed, size_t numElements)
{
    float err = 0;

    for(int i = 0; i < numElements; ++i)
    {
        float fv = src[i];

        fv *= fv;

        float e = processed[i] - fv;

        err += e * e;
    }    

    return err;
}

//-------------------------------------------------------------

int main()
{
    const size_t NUM_ELEMENTS = 1024 * 1024;
    const size_t BUF_SIZE = NUM_ELEMENTS * sizeof(float);

    size_t workSize = NUM_ELEMENTS;

    float* srcData = (float*)malloc(BUF_SIZE);

    fillSrcDataRandom(srcData, NUM_ELEMENTS);

        //

    ClPlayground cl;

        //

    cl_int errcode;

    cl_device_id dev;

    cl_context devco = NULL;
    cl_command_queue cmd = NULL;
    
    cl_mem buffer = NULL;

    cl_program program = NULL;
    cl_kernel kernelDoSome = NULL;

    void* mappedBuffer = NULL;
        
        //

    struct timeval before, after;

        //

    if(!cl.Init())
    {
        printf("Error CL init\n");

        goto cleanup;
    }

    printf("CL init OK (CPUs: %u, GPUs: %u)\n", cl.cpuDevicesCount, cl.gpuDevicesCount);
        
        //

    dev = cl.gpuDevices[0];

    devco = clCreateContext(NULL, 1, &dev, cl_notify, NULL, &errcode);

    if(devco == NULL)
    {
        printf("Error context creation\n");

        goto cleanup;        
    }

    cmd = clCreateCommandQueue(devco, dev, 0, &errcode);

    if(cmd == NULL)
    {
        printf("Error command queue creation\n");

        goto cleanup;                
    }
        //
    
    buffer = clCreateBuffer
                (
                    devco, 
                    CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
                    BUF_SIZE,
                    srcData,
                    &errcode
                );
    
    if(buffer == NULL)
    {
        printf("Error buffer creation\n");

        goto cleanup;
    }

        //

    program = cl.PrepareProgram(dev, devco, "#include \"kernel1.cl\"");

    if(!cl.clOk)
    {
        printf("Error program creation\n");

        goto cleanup;        
    }

    kernelDoSome = clCreateKernel(program, "doSome", &errcode);

    if(kernelDoSome == NULL)
    {
        printf("Error kernel creation\n");

        goto cleanup;                
    }

        //

    gettimeofday(&before, NULL);

    errcode = clSetKernelArg(kernelDoSome, 0, sizeof(cl_mem), &buffer);

    if(errcode != CL_SUCCESS)
    {
        printf("Error kernel arg set\n");

        goto cleanup;                        
    }
    
    errcode = clEnqueueNDRangeKernel(cmd, kernelDoSome, 1, NULL, &workSize, NULL, 0, NULL, NULL);

    if(errcode != CL_SUCCESS)
    {
        printf("Error kernel execution\n");

        goto cleanup;                        
    }

        //

    mappedBuffer = clEnqueueMapBuffer(cmd, buffer, CL_TRUE, CL_MAP_READ, 0, BUF_SIZE, 0, NULL, NULL, &errcode);

    gettimeofday(&after, NULL);

    if(mappedBuffer == NULL)
    {
        printf("Error read-back buffer map\n");

        goto cleanup;                                
    }

    printf("Execution time: %u ms\n", timeDifference(&before, &after) / 1000);

    printf("Err tested: %f\n", calcErr(srcData, (float*)mappedBuffer, NUM_ELEMENTS));

    clEnqueueUnmapMemObject(cmd, buffer, mappedBuffer, 0, NULL, NULL);

cleanup:

    clReleaseKernel(kernelDoSome);
    clReleaseProgram(program);
    clReleaseMemObject(buffer);
    clReleaseCommandQueue(cmd);
    clReleaseContext(devco);    

    cl.Dispose();

    free(srcData);

    printf("Bye.\n");    
    return 0;    
}

//-------------------------------------------------------------
