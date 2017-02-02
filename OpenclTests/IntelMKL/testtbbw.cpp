#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <windows.h>

#include <tbb/tbb.h>

//#include "../threadpool.h"
//#include "../threadpool_w.h"
//#include "../threadpool_we.h"
//#include "../threadpool_pw.h"
#include "../threadpool_p.h"

//-------------------------------------------------------------
// aligned memory allocation cross-stuff

#include <stdlib.h>

#ifdef _WIN32

#include <malloc.h>

#define aligned_alloc(alignment, size) _aligned_malloc((size), (alignment))
#define aligned_free _aligned_free
#define align_as(alignment) __declspec(align(alignment))

#else

void* aligned_alloc(size_t alignment, size_t size);

void* aligned_alloc(size_t alignment, size_t size)
{
    void* p = NULL;
    posix_memalign (&p, alignment, size);
    return p;
}

#define aligned_free free
#define align_as(alignment) __attribute__((aligned((alignment))))

#endif

//-------------------------------------------------------------

#include <xmmintrin.h>

using namespace tbb;

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

float getRandom(float amplitude)
{
    return floor(amplitude * ((float)rand() / (float)RAND_MAX));
}

//-------------------------------------------------------------

class FillRandom
{
    float *const mem_buffer;

    float mem_amplitude;

public:

    FillRandom(float* buffer, float amplitude) : 
        mem_buffer(buffer), 
        mem_amplitude(amplitude)
    {}

    /*
    void operator()(const blocked_range<size_t>& r) const
    {
        align_as(16) float buf[4]; 
        __m128* pBuf = (__m128*)buf;
        
        const size_t blockSize = 4;

        float* p = mem_buffer;
        float a = mem_amplitude;

        for(size_t i = r.begin(); i != r.end(); ++i)
        {
            buf[0] = getRandom(a);
            buf[1] = getRandom(a);
            buf[2] = getRandom(a);
            buf[3] = getRandom(a);

            _mm_store_ps(p + i * blockSize, *pBuf);
        }
    }
    */
    //*
    void operator()(const blocked_range<size_t>& r) const
    {
        float* p = mem_buffer;
        float a = mem_amplitude;

        for(size_t i = r.begin(); i != r.end(); ++i)
        {
            p[i] = getRandom(a);
        }
    } 
    //*/   
};

//-------------------------------------------------------------

#if 0

struct FillRandomTp : public ThreadPool
{   
    struct TaskItem
    {
        int code;                
        float* buffer;
        float amplitude;

        size_t offset;
        size_t count;
    };

    TaskItem* taskItems;

    static const int STOP = 0;
    static const int FILL_RANDOM = 1;

    FillRandomTp() : 
        taskItems(NULL)
    {}

    bool Init()
    {        
        if(!ThreadPool::Init(get_ncpu()))        
        {
            return false;
        }        

        taskItems = (TaskItem*)malloc(ThreadPool::capacity * sizeof(TaskItem));

        return true;
    }

    void Dispose()
    {
        chargeStop();
        WaitResults();

        ThreadPool::Dispose();    

        free(taskItems);
    }

    void FillRandom(float* buffer, float amplitude, size_t count)
    {
        chargeFillRandom(buffer, amplitude, count);
        WaitResults();
        Recharge();
    }

protected:

    virtual bool ProcessTask(int index)
    {
        int code = taskItems[index].code;

        if(code == STOP)
        {
            return false;
        }
        else //if(code == FILL_RANDOM)
        {
            kernelFillRandom(index);
        }

        return true;
    }

private:

    void chargeStop()
    {
        for(size_t i = 0; i < ThreadPool::capacity; ++i)
        {
            taskItems[i].code = STOP;            
        }        
    }    

    void chargeFillRandom(float* buffer, float amplitude, size_t count)
    {
        size_t blockSize = count / ThreadPool::capacity;

        size_t totalWorkSize = blockSize * ThreadPool::capacity;
        
        for(size_t i = 0; i < ThreadPool::capacity; ++i)
        {
            taskItems[i].code = FILL_RANDOM;
            taskItems[i].buffer = buffer;            
            taskItems[i].amplitude = amplitude;            
            
            taskItems[i].offset = blockSize * i;
            taskItems[i].count = blockSize + ((i == ThreadPool::capacity - 1) ? (count - totalWorkSize) : 0);
        }        
    }        

    /*
    void kernelFillRandom(int index)
    {
        float* buffer = taskItems[index].buffer;
        float amplitude = taskItems[index].amplitude;

        size_t offset = taskItems[index].offset;
        size_t count = taskItems[index].count;

        const size_t blockSize = 4;

        align_as(16) float buf[4]; 
        __m128* pBuf = (__m128*)buf;

        for(size_t i = offset; i < count; i += blockSize)
        {
            buf[0] = getRandom(amplitude);
            buf[1] = getRandom(amplitude);
            buf[2] = getRandom(amplitude);
            buf[3] = getRandom(amplitude);

            _mm_store_ps(buffer + offset, *pBuf);
        }        
    }
    */
    //*
    void kernelFillRandom(int index)
    {
        float* buffer = taskItems[index].buffer;
        float amplitude = taskItems[index].amplitude;

        size_t offset = taskItems[index].offset;
        size_t count = taskItems[index].count;

        size_t stop = offset + count;

        for(size_t i = offset; i < stop; ++i)
        {
            buffer[i] = getRandom(amplitude);
        }        
    }
    //*/    
};

#endif

struct FillRandomTp : public ThreadPool
{   
    float* memBuffer;
    float memAmplitude;        

    bool Init()
    {
        return ThreadPool::Init(get_ncpu());
    }

    void Dispose()
    {
        ThreadPool::stop = true;
        WaitResults();

        ThreadPool::Dispose();    
    }

    void FillRandom(float* buffer, float amplitude, size_t count)
    {
        chargeFillRandom(buffer, amplitude, count);
        WaitResults();
        Recharge();
    }

private:

    virtual void ProcessTask(/*int index,*/ std::pair<size_t, size_t>& item)
    {
        float* buffer = memBuffer;
        float amplitude = memAmplitude;

        for(size_t i = item.first; i < item.second; ++i)
        {
            buffer[i] = getRandom(amplitude);
        }        
    }

    void chargeFillRandom(float* buffer, float amplitude, size_t count)
    {
        const size_t grainsCountPerProcessor = 256;

        size_t grainsCount = ThreadPool::capacity * grainsCountPerProcessor;

        size_t blockSize = count / grainsCount;

        size_t acc = 0;
        size_t next = acc + blockSize;

        for(size_t i = 0; i < grainsCount - 1; ++i, acc = next, next += blockSize)
        {            
            ThreadPool::items.push(std::pair<size_t, size_t>(acc, next));            
        }

        ThreadPool::items.push(std::pair<size_t, size_t>(acc, count));
        
        memBuffer = buffer;
        memAmplitude = amplitude;
    }        
};

//-------------------------------------------------------------

int main()
{
    size_t dim = 64 * 1000 * 1000;

    float* buffer = (float*)aligned_alloc(16, dim * sizeof(float));

    FILETIME before, after;

    FillRandomTp frtp;
    
    if(!frtp.Init())
    {
        printf("Error tp init\n");
        goto cleanup;    
    }

    GetSystemTimeAsFileTime(&before);
    srand(before.dwLowDateTime);

    GetSystemTimeAsFileTime(&before);
    for(size_t i = 0; i < 20; ++i)
    //parallel_for(blocked_range<size_t>(0, dim/4), FillRandom(buffer, 1000));         
    //parallel_for(blocked_range<size_t>(0, dim), FillRandom(buffer, 1000));         
    /*
    parallel_for(size_t(0), dim, [=](size_t i)
    {
        buffer[i] = getRandom(1000);
    });
    */
    frtp.FillRandom(buffer, 1000, dim);
    GetSystemTimeAsFileTime(&after);

    printf("Execution time: %u ms.\n", timeDifference(&before, &after) / 1000);

cleanup:

    frtp.Dispose();

    aligned_free(buffer);
    
    printf("Bye.\n");

    return 0;
}

//-------------------------------------------------------------
