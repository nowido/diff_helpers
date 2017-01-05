#ifndef OCLHELPERS_H
#define OCLHELPERS_H

#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

//-------------------------------------------------------------

cl_uint getClPlatformsCount()
{
    cl_uint num = 0;

    clGetPlatformIDs(1, NULL, &num);    

    return num;
}

//-------------------------------------------------------------

cl_uint getClPlatformIds(cl_platform_id *plids, cl_uint count)
{
    return clGetPlatformIDs(count, plids, NULL);
}

//-------------------------------------------------------------

struct ClPlatformInfo
{
    static const cl_uint NumberOfEntries = 5;

    static const int mapProfile     = 0;
    static const int mapVersion     = 1;
    static const int mapName        = 2;
    static const int mapVendor      = 3;
    static const int mapExtensions  = 4;

    char* entries[ClPlatformInfo::NumberOfEntries];

        //

    void Init()
    {
        memset(entries, 0, ClPlatformInfo::NumberOfEntries * sizeof(char*));
    }

    void Dispose()
    {
        for(int i = 0; i < ClPlatformInfo::NumberOfEntries; ++i)
        {
            free(entries[i]);    
        }
    }
};

//-------------------------------------------------------------

cl_uint getClPlatformInfo
        (
            cl_platform_id platform, 
            ClPlatformInfo *info
        )
{
    size_t size = 0;

    cl_uint res;

    static const cl_platform_info query[ClPlatformInfo::NumberOfEntries] = 
    {
        CL_PLATFORM_PROFILE,
        CL_PLATFORM_VERSION,
        CL_PLATFORM_NAME,
        CL_PLATFORM_VENDOR,
        CL_PLATFORM_EXTENSIONS    
    };

    static const int mapping[ClPlatformInfo::NumberOfEntries] =
    {
        ClPlatformInfo::mapProfile,
        ClPlatformInfo::mapVersion,
        ClPlatformInfo::mapName,
        ClPlatformInfo::mapVendor,
        ClPlatformInfo::mapExtensions
    };
    
    for(int i = 0; i < ClPlatformInfo::NumberOfEntries; ++i)
    {        
        res = clGetPlatformInfo(platform, query[i], 0, NULL, &size);

        if(res != CL_SUCCESS)
        {
            return res;
        }
        
        char* infoBuffer = *(info->entries + mapping[i]) = (char*)malloc(size);
        
        res = clGetPlatformInfo(platform, query[i], size, infoBuffer, NULL);

        if(res != CL_SUCCESS)
        {
            return res;
        } 
    }

    return res;
}

//-------------------------------------------------------------

cl_uint getClDevicesCount
        (
            cl_platform_id platform, 
            cl_device_type devType
        )
{
    cl_uint num = 0;

    clGetDeviceIDs(platform, devType, 0, NULL, &num);    

    return num;
}

//-------------------------------------------------------------

cl_uint getClDeviceIds
        (
            cl_platform_id platform, 
            cl_device_type devType, 
            cl_device_id *devices, 
            cl_uint count
        )
{
    return clGetDeviceIDs(platform, devType, count, devices, NULL);    
}

//-------------------------------------------------------------

struct ClDeviceInfo
{
    static const cl_uint NumberOfEntries = 72;

    static const int mapType                        = 0;
    static const int mapVendorId                    = 1;
    static const int mapMaxComputeUnits             = 2;
    static const int mapMaxWorkItemDimensions       = 3;
    static const int mapMaxWorkItemSizes            = 4;
    static const int mapMaxWorkGroupSize            = 5;
    static const int mapPreferredVectorWidthChar    = 6;
    static const int mapPreferredVectorWidthShort   = 7;
    static const int mapPreferredVectorWidthInt     = 8;
    static const int mapPreferredVectorWidthLong    = 9;
    static const int mapPreferredVectorWidthFloat   = 10;
    static const int mapPreferredVectorWidthDouble  = 11;
    static const int mapPreferredVectorWidthHalf    = 12;
    static const int mapNativeVectorWidthChar       = 13;
    static const int mapNativeVectorWidthShort      = 14;
    static const int mapNativeVectorWidthInt        = 15;
    static const int mapNativeVectorWidthLong       = 16;
    static const int mapNativeVectorWidthFloat      = 17;
    static const int mapNativeVectorWidthDouble     = 18;
    static const int mapNativeVectorWidthHalf       = 19;
    static const int mapMaxClockFrequency           = 20;
    static const int mapAddressBits                 = 21;
    static const int mapMaxMemAllocSize             = 22;
    static const int mapImageSupport                = 23;
    static const int mapMaxReadImageArgs            = 24;
    static const int mapMaxWriteImageArgs           = 25;
    static const int mapImage2DMaxWidth             = 26;
    static const int mapImage2DMaxHeight            = 27;
    static const int mapImage3DMaxWidth             = 28;
    static const int mapImage3DMaxHeight            = 29;
    static const int mapImage3DMaxDepth             = 30;
    static const int mapImageMaxBufferSize          = 31;
    static const int mapImageMaxArraySize           = 32;
    static const int mapMaxSamplers                 = 33;
    static const int mapMaxParameterSize            = 34;
    static const int mapMemBaseAddrAlign            = 35;
    static const int mapSingleFpConfig              = 36;
    static const int mapDoubleFpConfig              = 37;
    static const int mapGlobalMemCacheType          = 38;
    static const int mapGlobalMemCachelineSize      = 39;
    static const int mapGlobalMemCacheSize          = 40;
    static const int mapGlobalMemSize               = 41;
    static const int mapMaxConstantBufferSize       = 42;
    static const int mapMaxConstantArgs             = 43;
    static const int mapLocalMemType                = 44;
    static const int mapLocalMemSize                = 45;
    static const int mapErrorCorrectionSupport      = 46;
    static const int mapHostUnifiedMemory           = 47;
    static const int mapProfilingTimerResolution    = 48;
    static const int mapEndianLittle                = 49;
    static const int mapAvailable                   = 50;
    static const int mapCompilerAvailable           = 51;
    static const int mapLinkerAvailable             = 52;
    static const int mapExecutionCapabilities       = 53;
    static const int mapQueueProperties             = 54;
    static const int mapBuiltInKernels              = 55;
    static const int mapPlatform                    = 56;
    static const int mapName                        = 57;
    static const int mapVendor                      = 58;
    static const int mapDriverVersion               = 59;
    static const int mapProfile                     = 60;
    static const int mapVersion                     = 61;
    static const int mapOpenclCVersion              = 62;
    static const int mapExtensions                  = 63;
    static const int mapPrintfBufferSize            = 64;
    static const int mapPreferredInteropUserSync    = 65;
    static const int mapParentDevice                = 66;
    static const int mapPartitionMaxSubDevices      = 67;
    static const int mapPartitionProperties         = 68;
    static const int mapPartitionAffinityDomain     = 69;
    static const int mapPartitionType               = 70;
    static const int mapReferenceCount              = 71; 

    void* entries[ClDeviceInfo::NumberOfEntries];
    
        //

    void Init()
    {
        memset(entries, 0, ClDeviceInfo::NumberOfEntries * sizeof(void*));
    }

    void Dispose()
    {
        for(int i = 0; i < ClDeviceInfo::NumberOfEntries; ++i)
        {
            free(entries[i]);    
        }
    }
};

//-------------------------------------------------------------

cl_uint getClDeviceInfo
        (
            cl_device_id device, 
            ClDeviceInfo *info
        )
{
    size_t size = 0;

    cl_uint res;

    static const cl_device_info query[ClDeviceInfo::NumberOfEntries] = 
    {
        CL_DEVICE_TYPE,                             // 0
        CL_DEVICE_VENDOR_ID,                        // 1
        CL_DEVICE_MAX_COMPUTE_UNITS,                // 2
        CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,         // 3
        CL_DEVICE_MAX_WORK_ITEM_SIZES,              // 4
        CL_DEVICE_MAX_WORK_GROUP_SIZE,              // 5
        CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR,      // 6
        CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT,     // 7
        CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT,       // 8
        CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG,      // 9
        CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,     // 10
        CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,    // 11
        CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF,      // 12
        CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR,         // 13        
        CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT,        // 14
        CL_DEVICE_NATIVE_VECTOR_WIDTH_INT,          // 15
        CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG,         // 16
        CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT,        // 17
        CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE,       // 18
        CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF,         // 19
        CL_DEVICE_MAX_CLOCK_FREQUENCY,              // 20
        CL_DEVICE_ADDRESS_BITS,                     // 21
        CL_DEVICE_MAX_MEM_ALLOC_SIZE,               // 22
        CL_DEVICE_IMAGE_SUPPORT,                    // 23
        CL_DEVICE_MAX_READ_IMAGE_ARGS,              // 24
        CL_DEVICE_MAX_WRITE_IMAGE_ARGS,             // 25
        CL_DEVICE_IMAGE2D_MAX_WIDTH,                // 26
        CL_DEVICE_IMAGE2D_MAX_HEIGHT,               // 27
        CL_DEVICE_IMAGE3D_MAX_WIDTH,                // 28
        CL_DEVICE_IMAGE3D_MAX_HEIGHT,               // 29
        CL_DEVICE_IMAGE3D_MAX_DEPTH,                // 30
        CL_DEVICE_IMAGE_MAX_BUFFER_SIZE,            // 31
        CL_DEVICE_IMAGE_MAX_ARRAY_SIZE,             // 32
        CL_DEVICE_MAX_SAMPLERS,                     // 33
        CL_DEVICE_MAX_PARAMETER_SIZE,               // 34
        CL_DEVICE_MEM_BASE_ADDR_ALIGN,              // 35
        CL_DEVICE_SINGLE_FP_CONFIG,                 // 36
        CL_DEVICE_DOUBLE_FP_CONFIG,                 // 37
        CL_DEVICE_GLOBAL_MEM_CACHE_TYPE,            // 38
        CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,        // 39
        CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,            // 40
        CL_DEVICE_GLOBAL_MEM_SIZE,                  // 41
        CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,         // 42
        CL_DEVICE_MAX_CONSTANT_ARGS,                // 43
        CL_DEVICE_LOCAL_MEM_TYPE,                   // 44
        CL_DEVICE_LOCAL_MEM_SIZE,                   // 45
        CL_DEVICE_ERROR_CORRECTION_SUPPORT,         // 46
        CL_DEVICE_HOST_UNIFIED_MEMORY,              // 47
        CL_DEVICE_PROFILING_TIMER_RESOLUTION,       // 48
        CL_DEVICE_ENDIAN_LITTLE,                    // 49
        CL_DEVICE_AVAILABLE,                        // 50
        CL_DEVICE_COMPILER_AVAILABLE,               // 51
        CL_DEVICE_LINKER_AVAILABLE,                 // 52
        CL_DEVICE_EXECUTION_CAPABILITIES,           // 53
        CL_DEVICE_QUEUE_PROPERTIES,                 // 54
        CL_DEVICE_BUILT_IN_KERNELS,                 // 55
        CL_DEVICE_PLATFORM,                         // 56
        CL_DEVICE_NAME,                             // 57
        CL_DEVICE_VENDOR,                           // 58
        CL_DRIVER_VERSION,                          // 59
        CL_DEVICE_PROFILE,                          // 60
        CL_DEVICE_VERSION,                          // 61
        CL_DEVICE_OPENCL_C_VERSION,                 // 62
        CL_DEVICE_EXTENSIONS,                       // 63
        CL_DEVICE_PRINTF_BUFFER_SIZE,               // 64
        CL_DEVICE_PREFERRED_INTEROP_USER_SYNC,      // 65
        CL_DEVICE_PARENT_DEVICE,                    // 66
        CL_DEVICE_PARTITION_MAX_SUB_DEVICES,        // 67
        CL_DEVICE_PARTITION_PROPERTIES,             // 68
        CL_DEVICE_PARTITION_AFFINITY_DOMAIN,        // 69
        CL_DEVICE_PARTITION_TYPE,                   // 70
        CL_DEVICE_REFERENCE_COUNT                   // 71
    };

    static const int mapping[ClDeviceInfo::NumberOfEntries] =
    {
        ClDeviceInfo::mapType,    
        ClDeviceInfo::mapVendorId,
        ClDeviceInfo::mapMaxComputeUnits,
        ClDeviceInfo::mapMaxWorkItemDimensions,
        ClDeviceInfo::mapMaxWorkItemSizes,
        ClDeviceInfo::mapMaxWorkGroupSize,
        ClDeviceInfo::mapPreferredVectorWidthChar,
        ClDeviceInfo::mapPreferredVectorWidthShort,
        ClDeviceInfo::mapPreferredVectorWidthInt,
        ClDeviceInfo::mapPreferredVectorWidthLong,
        ClDeviceInfo::mapPreferredVectorWidthFloat,
        ClDeviceInfo::mapPreferredVectorWidthDouble,
        ClDeviceInfo::mapPreferredVectorWidthHalf,
        ClDeviceInfo::mapNativeVectorWidthChar,
        ClDeviceInfo::mapNativeVectorWidthShort,
        ClDeviceInfo::mapNativeVectorWidthInt,
        ClDeviceInfo::mapNativeVectorWidthLong,
        ClDeviceInfo::mapNativeVectorWidthFloat,
        ClDeviceInfo::mapNativeVectorWidthDouble,
        ClDeviceInfo::mapNativeVectorWidthHalf,
        ClDeviceInfo::mapMaxClockFrequency,
        ClDeviceInfo::mapAddressBits,
        ClDeviceInfo::mapMaxMemAllocSize,
        ClDeviceInfo::mapImageSupport,
        ClDeviceInfo::mapMaxReadImageArgs,
        ClDeviceInfo::mapMaxWriteImageArgs,
        ClDeviceInfo::mapImage2DMaxWidth,
        ClDeviceInfo::mapImage2DMaxHeight,
        ClDeviceInfo::mapImage3DMaxWidth,
        ClDeviceInfo::mapImage3DMaxHeight,
        ClDeviceInfo::mapImage3DMaxDepth,
        ClDeviceInfo::mapImageMaxBufferSize,
        ClDeviceInfo::mapImageMaxArraySize,
        ClDeviceInfo::mapMaxSamplers,
        ClDeviceInfo::mapMaxParameterSize,
        ClDeviceInfo::mapMemBaseAddrAlign,
        ClDeviceInfo::mapSingleFpConfig,
        ClDeviceInfo::mapDoubleFpConfig,
        ClDeviceInfo::mapGlobalMemCacheType,
        ClDeviceInfo::mapGlobalMemCachelineSize,
        ClDeviceInfo::mapGlobalMemCacheSize,
        ClDeviceInfo::mapGlobalMemSize,
        ClDeviceInfo::mapMaxConstantBufferSize,
        ClDeviceInfo::mapMaxConstantArgs,
        ClDeviceInfo::mapLocalMemType,
        ClDeviceInfo::mapLocalMemSize,
        ClDeviceInfo::mapErrorCorrectionSupport,
        ClDeviceInfo::mapHostUnifiedMemory,
        ClDeviceInfo::mapProfilingTimerResolution,
        ClDeviceInfo::mapEndianLittle,
        ClDeviceInfo::mapAvailable,
        ClDeviceInfo::mapCompilerAvailable,
        ClDeviceInfo::mapLinkerAvailable,
        ClDeviceInfo::mapExecutionCapabilities,
        ClDeviceInfo::mapQueueProperties,
        ClDeviceInfo::mapBuiltInKernels,
        ClDeviceInfo::mapPlatform,
        ClDeviceInfo::mapName,
        ClDeviceInfo::mapVendor,
        ClDeviceInfo::mapDriverVersion,
        ClDeviceInfo::mapProfile,
        ClDeviceInfo::mapVersion,
        ClDeviceInfo::mapOpenclCVersion,
        ClDeviceInfo::mapExtensions,
        ClDeviceInfo::mapPrintfBufferSize,
        ClDeviceInfo::mapPreferredInteropUserSync,
        ClDeviceInfo::mapParentDevice,
        ClDeviceInfo::mapPartitionMaxSubDevices,
        ClDeviceInfo::mapPartitionProperties,
        ClDeviceInfo::mapPartitionAffinityDomain,
        ClDeviceInfo::mapPartitionType,
        ClDeviceInfo::mapReferenceCount
    };

    for(int i = 0; i < ClDeviceInfo::NumberOfEntries; ++i)
    {        
        res = clGetDeviceInfo(device, query[i], 0, NULL, &size);

        if(res != CL_SUCCESS)
        {
            return res;
        }
        
        void* infoBuffer = *(info->entries + mapping[i]) = malloc(size);
        
        res = clGetDeviceInfo(device, query[i], size, infoBuffer, NULL);

        if(res != CL_SUCCESS)
        {
            return res;
        } 
    }

    return res;    
}

//-------------------------------------------------------------

#endif
