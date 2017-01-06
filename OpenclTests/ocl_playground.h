#ifndef OCL_PLAYGROUND_H
#define OCL_PLAYGROUND_H

#include "oclhelpers.h"

//-------------------------------------------------------------

struct ClPlayground
{
    bool clOk;

    cl_uint platformsCount;
    cl_platform_id *platformIds;

    cl_uint cpuDevicesCount;
    cl_device_id *cpuDevices;

    cl_uint gpuDevicesCount;
    cl_device_id *gpuDevices;

    char* buildLog;

    /////////////////////////////////////////
    bool Init(cl_uint platformIndex = 0)
    {
        platformIds = NULL;
        cpuDevices = NULL;
        gpuDevices = NULL;

        cpuDevicesCount = 0;
        gpuDevicesCount = 0;

        buildLog = NULL;

        clOk = false;
        
        platformsCount = getClPlatformsCount();

        if(platformsCount > 0)
        {
            platformIds = (cl_platform_id*)malloc(sizeof(cl_platform_id) * platformsCount);

            clOk = (getClPlatformIds(platformIds, platformsCount) == CL_SUCCESS);

            if(clOk)
            {
                cl_platform_id usePlatform = platformIds[platformIndex];

                cpuDevicesCount = getClDevicesCount(usePlatform, CL_DEVICE_TYPE_CPU);
                
                if(cpuDevicesCount > 0)
                {
                    cpuDevices = (cl_device_id*)malloc(sizeof(cl_device_id) * cpuDevicesCount);

                    clOk = (getClDeviceIds(usePlatform, CL_DEVICE_TYPE_CPU, cpuDevices, cpuDevicesCount) == CL_SUCCESS);
                }

                gpuDevicesCount = getClDevicesCount(usePlatform, CL_DEVICE_TYPE_GPU);

                if(gpuDevicesCount > 0)
                {
                    gpuDevices = (cl_device_id*)malloc(sizeof(cl_device_id) * gpuDevicesCount);

                    clOk = clOk && (getClDeviceIds(usePlatform, CL_DEVICE_TYPE_GPU, gpuDevices, gpuDevicesCount) == CL_SUCCESS);                    
                }
            }

            return clOk;
        }

        return clOk;
    }

    /////////////////////////////////////////
    cl_program PrepareProgram
                (
                    cl_device_id dev, 
                    cl_context ctx, 
                    const char* source, 
                    char* options = NULL
                )
    {
        clOk = false;

        cl_int errcode;

        cl_program program = clCreateProgramWithSource(ctx, 1, &source, NULL, &errcode);
        
        if(errcode != CL_SUCCESS)
        {
            return program;
        }

        errcode = clBuildProgram(program, 1, &dev, options, NULL, NULL);

        if(errcode == CL_SUCCESS)
        {
            clOk = true;
        }
        else
        {
            size_t logSize = 0;

            clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);

            if(logSize > 0)
            {
                free(buildLog);  

                buildLog = (char*)malloc(logSize);  

                clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, logSize, buildLog, NULL);
            }
        }

        return program;
    }

    /////////////////////////////////////////
    void Dispose()
    {
        free(platformIds);
        free(cpuDevices);
        free(gpuDevices);
        free(buildLog);
    }
};

//-------------------------------------------------------------

#endif
