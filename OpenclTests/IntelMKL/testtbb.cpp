#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <sys/time.h>
#include <unistd.h>

#include <tbb/tbb.h>

#include "../threadpool_p.h"

using namespace tbb;

//-------------------------------------------------------------

unsigned int timeDifference(struct timeval *before, struct timeval *after)
{
    return ((after->tv_sec * 1000000 + after->tv_usec) - (before->tv_sec * 1000000 + before->tv_usec));
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

    void operator()(const blocked_range<size_t>& r) const
    {
        float* p = mem_buffer;
        float a = mem_amplitude;

        for(size_t i = r.begin(); i != r.end(); ++i)
        {
            p[i] = getRandom(a);
        }
    }
};

//-------------------------------------------------------------

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
        const size_t grainsCountPerProcessor = 128;

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

    float* buffer = (float*)malloc(dim * sizeof(float));

    struct timeval before, after;

    FillRandomTp frtp;
    
    if(!frtp.Init())
    {
        printf("Error tp init\n");
        goto cleanup;    
    }

    gettimeofday(&before, NULL);
    srand(before.tv_usec);

    gettimeofday(&before, NULL);
    for(size_t i = 0; i < 20; ++i)
    {
        //static affinity_partitioner ap;
        //parallel_for(blocked_range<size_t>(0, dim), FillRandom(buffer, 1000), ap);    

        //parallel_for(blocked_range<size_t>(0, dim, 100000), FillRandom(buffer, 1000), simple_partitioner());    
        //parallel_for(blocked_range<size_t>(0, dim), FillRandom(buffer, 1000));    
        /*
        parallel_for(size_t(0), dim, [=](size_t i)
        {
            buffer[i] = getRandom(1000);
        });
        */
        frtp.FillRandom(buffer, 1000, dim);
    }
    gettimeofday(&after, NULL);

    printf("Execution time: %u ms.\n", timeDifference(&before, &after) / 1000);

cleanup:

    frtp.Dispose();

    free(buffer);
    
    printf("Bye.\n");

    return 0;
}

//-------------------------------------------------------------
