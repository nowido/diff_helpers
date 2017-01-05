#include <stdio.h>
#include <stdlib.h>

#include "oclhelpers.h"

//-------------------------------------------------------------

void printInfo(ClDeviceInfo *info)
{
    printf("Name: %s\n", (char*)(info->entries[ClDeviceInfo::mapName]));
    printf("Max clock frequency: %u MHz\n", *(cl_uint*)(info->entries[ClDeviceInfo::mapMaxClockFrequency]));
    printf("Address bits: %u\n", *(cl_uint*)(info->entries[ClDeviceInfo::mapAddressBits]));
    printf("Max compute units: %u\n", *(cl_uint*)(info->entries[ClDeviceInfo::mapMaxComputeUnits]));
    printf("Max work item dimensions: %u\n", *(cl_uint*)(info->entries[ClDeviceInfo::mapMaxWorkItemDimensions]));    
    printf("Max work group size: %u\n", *(cl_uint*)(info->entries[ClDeviceInfo::mapMaxWorkGroupSize]));    
    printf("Global mem size: %llu\n", *(cl_ulong*)(info->entries[ClDeviceInfo::mapGlobalMemSize]));    
    printf("Local mem size: %llu\n", *(cl_ulong*)(info->entries[ClDeviceInfo::mapLocalMemSize])); 
    printf("Host unified memory: %u\n", *(cl_bool*)(info->entries[ClDeviceInfo::mapHostUnifiedMemory]));
    printf("Image support: %u\n", *(cl_bool*)(info->entries[ClDeviceInfo::mapImageSupport]));    
    printf("Max read image args: %u\n", *(cl_uint*)(info->entries[ClDeviceInfo::mapMaxReadImageArgs])); 
    printf("Max write image args: %u\n", *(cl_uint*)(info->entries[ClDeviceInfo::mapMaxWriteImageArgs])); 
    printf("Image2D max width: %lu\n", *(size_t*)(info->entries[ClDeviceInfo::mapImage2DMaxWidth])); 
    printf("Image2D max height: %lu\n", *(size_t*)(info->entries[ClDeviceInfo::mapImage2DMaxHeight])); 
    printf("Preferred vector width (double): %u\n", *(cl_uint*)(info->entries[ClDeviceInfo::mapPreferredVectorWidthDouble]));
    printf("Native vector width (double): %u\n", *(cl_uint*)(info->entries[ClDeviceInfo::mapNativeVectorWidthDouble])); 
    printf("Preferred vector width (float): %u\n", *(cl_uint*)(info->entries[ClDeviceInfo::mapPreferredVectorWidthFloat]));
    printf("Native vector width (float): %u\n", *(cl_uint*)(info->entries[ClDeviceInfo::mapNativeVectorWidthFloat]));     
}

//-------------------------------------------------------------

int main()
{
    const int PlatformIndex = 0;

    const int CPUIndex = 0;
    const int GPUIndex = 0;

        // static things, no cleanup needed

    cl_uint platformsCount = 0;
    cl_platform_id usePlatform = 0;

    cl_uint cpuDevicesCount = 0;
    cl_device_id useCpuDevice = 0;

    cl_uint gpuDevicesCount = 0;
    cl_device_id useGpuDevice = 0;

        // things to cleanup later

    cl_platform_id *platformIds = NULL;

    cl_device_id *cpuDevices = NULL;
    cl_device_id *gpuDevices = NULL;

    ClPlatformInfo platformInfo;
    platformInfo.Init();

    ClDeviceInfo cpuDeviceInfo;
    cpuDeviceInfo.Init();

    ClDeviceInfo gpuDeviceInfo;
    gpuDeviceInfo.Init();

        //

    platformsCount = getClPlatformsCount();

    printf("Number of OpenCL platforms: %d\n", platformsCount);

    if(platformsCount == 0)
    {
        printf("No OpenCL platforms.\n");

        goto cleanup;
    }

    platformIds = (cl_platform_id*)malloc(sizeof(cl_platform_id) * platformsCount);

    if(getClPlatformIds(platformIds, platformsCount) != CL_SUCCESS)
    {
        printf("Error getClPlatformIds.\n");

        goto cleanup;
    }

    usePlatform = platformIds[PlatformIndex];

    if(getClPlatformInfo(usePlatform, &platformInfo) != CL_SUCCESS)
    {
        printf("Error getClPlatformInfo.\n");

        goto cleanup;        
    }
    else
    {
        printf("Profile: %s\n", platformInfo.entries[ClPlatformInfo::mapProfile]);        
        printf("Version: %s\n", platformInfo.entries[ClPlatformInfo::mapVersion]);        
        printf("Name: %s\n", platformInfo.entries[ClPlatformInfo::mapName]);
        printf("Vendor: %s\n", platformInfo.entries[ClPlatformInfo::mapVendor]);
        printf("Extensions: %s\n", platformInfo.entries[ClPlatformInfo::mapExtensions]);        
    }

    cpuDevicesCount = getClDevicesCount(usePlatform, CL_DEVICE_TYPE_CPU);
    gpuDevicesCount = getClDevicesCount(usePlatform, CL_DEVICE_TYPE_GPU);

    printf("CPUs: %u, GPUs: %u\n", cpuDevicesCount, gpuDevicesCount);

    cpuDevices = (cl_device_id*)malloc(sizeof(cl_device_id) * cpuDevicesCount);
    gpuDevices = (cl_device_id*)malloc(sizeof(cl_device_id) * gpuDevicesCount);

    if(getClDeviceIds(usePlatform, CL_DEVICE_TYPE_CPU, cpuDevices, cpuDevicesCount) != CL_SUCCESS)
    {
        printf("Error getClDeviceIds (CPU)");

        goto cleanup;
    }

    if(getClDeviceIds(usePlatform, CL_DEVICE_TYPE_GPU, gpuDevices, gpuDevicesCount) != CL_SUCCESS)
    {
        printf("Error getClDeviceIds (GPU)");

        goto cleanup;
    }

    useCpuDevice = cpuDevices[CPUIndex];
    useGpuDevice = gpuDevices[GPUIndex];

    if(getClDeviceInfo(useCpuDevice, &cpuDeviceInfo) != CL_SUCCESS)
    {
        printf("Error getClDeviceInfo (CPU)");

        goto cleanup;
    }
    else
    {
        printf("--- CPU\n");

        printInfo(&cpuDeviceInfo);
    }

    if(getClDeviceInfo(useGpuDevice, &gpuDeviceInfo) != CL_SUCCESS)
    {
        printf("Error getClDeviceInfo (GPU)");

        goto cleanup;
    }
    else
    {
        printf("--- GPU\n");

        printInfo(&gpuDeviceInfo);
    }

cleanup:

    free(platformIds);

    free(cpuDevices);
    free(gpuDevices);

    platformInfo.Dispose();

    cpuDeviceInfo.Dispose();
    gpuDeviceInfo.Dispose();

        //

    printf("Bye.\n");
    
    return 0;
}
