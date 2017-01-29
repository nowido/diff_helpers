#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <sys/time.h>
#include <unistd.h>

#include <tbb/tbb.h>

#include "../threadpool.h"

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
    struct TaskItem
    {
        int code;                
        float* buffer;
        float amplitude;
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
        for(size_t i = 0; i < ThreadPool::capacity; ++i)
        {
            taskItems[i].code = FILL_RANDOM;
            taskItems[i].buffer = buffer;            
            taskItems[i].amplitude = amplitude;            
            taskItems[i].count = count;            
        }        
    }        

    void kernelFillRandom(int index)
    {
        float* buffer = taskItems[index].buffer;
        float amplitude = taskItems[index].amplitude;
        size_t count = taskItems[index].count;

        for(size_t i = index; i < count; i += ThreadPool::capacity)
        {
            buffer[i] = getRandom(amplitude);
        }
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
    for(size_t i = 0; i < 10; ++i)
    //parallel_for(blocked_range<size_t>(0, dim), FillRandom(buffer, 1000));    
    /*
    parallel_for(size_t(0), dim, [=](size_t i)
    {
        buffer[i] = getRandom(1000);
    });
    */
    frtp.FillRandom(buffer, 1000, dim);
    gettimeofday(&after, NULL);

    printf("Execution time: %u ms.\n", timeDifference(&before, &after) / 1000);

cleanup:

    frtp.Dispose();

    free(buffer);
    
    printf("Bye.\n");

    return 0;
}

//-------------------------------------------------------------
