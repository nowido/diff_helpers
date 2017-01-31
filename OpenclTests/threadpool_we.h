#ifndef THREADPOOLWE_H
#define THREADPOOLWE_H

#ifdef _WIN32

#include <windows.h>

size_t get_ncpu()
{
    SYSTEM_INFO si;

    GetSystemInfo(&si);

    return si.dwNumberOfProcessors;
}

//-------------------------------------------------------------

struct ThreadPool;

struct ThreadArgsEGNUEPAFLL
{
    ThreadPool* pool;
    int index;    
};

DWORD GlobalThreadProcEGNUEPAFLL(LPVOID arg);

//-------------------------------------------------------------

struct ThreadPool
{
    size_t capacity;

    HANDLE* thandles;    
    DWORD* tids;

    HANDLE* tasks;
    HANDLE* readies;

    ThreadArgsEGNUEPAFLL* array_run_args;

    DWORD masterThreadId;

    /////////////////////////////////////////
    ThreadPool() : 
        thandles(NULL),
        tids(NULL),
        tasks(NULL),
        readies(NULL),
        array_run_args(NULL)
    {}

    /////////////////////////////////////////
    bool Init(size_t argCapacity)
    {
        capacity = argCapacity;

        masterThreadId = GetCurrentThreadId();

        thandles = (HANDLE*)malloc(capacity * sizeof(HANDLE));
        tids = (DWORD*)malloc(capacity * sizeof(DWORD));

        tasks = (HANDLE*)malloc(capacity * sizeof(HANDLE));
        readies = (HANDLE*)malloc(capacity * sizeof(HANDLE));

        array_run_args = (ThreadArgsEGNUEPAFLL*)malloc(capacity * sizeof(ThreadArgsEGNUEPAFLL));

        for(size_t i = 0; i < capacity; ++i)
        {
            tasks[i] = CreateEvent(NULL, FALSE, FALSE, NULL);
            readies[i] = CreateEvent(NULL, FALSE, FALSE, NULL);

            if((tasks[i] == NULL) || (readies[i] == NULL))
            {
                return false;
            }
        }        

        for(size_t i = 0; i < capacity; ++i)
        {
            array_run_args[i].pool = this;
            array_run_args[i].index = i;            
        }

        for(size_t i = 0; i < capacity; ++i)
        {            
            thandles[i] = CreateThread(NULL, 0, GlobalThreadProcEGNUEPAFLL, (LPVOID)&(array_run_args[i]), 0, tids + i);

            if(thandles[i] == NULL)
            {
                return false;
            }
        }

            // ensure threads are created and blocked waiting incoming tasks
        WaitForMultipleObjects(capacity, readies, TRUE, INFINITE);

        return true;
    }

    /////////////////////////////////////////
    void Dispose()
    {
        WaitForMultipleObjects(capacity, thandles, TRUE, INFINITE);

        for(size_t i = 0; i < capacity; ++i)
        {
            CloseHandle(thandles[i]);
            CloseHandle(tasks[i]);            
            CloseHandle(readies[i]);            
        }

        free(thandles);    
        free(tids);
        free(readies);
        free(tasks);    
        free(array_run_args);
    }
    
    /////////////////////////////////////////
    void ThreadProc(int index)
    {
        MSG msg;

        while(true)
        {
            WaitForSingleObject(tasks[index], INFINITE);

            bool res = ProcessTask(index);

            SetEvent(readies[index]);
            
            if(!res)
            {
                return;
            }                        
        }
    }

    /////////////////////////////////////////
    void WaitResults()
    {
            // charge task to all worker threads
            
        for(size_t i = 0; i < capacity; ++i)
        {
            SetEvent(tasks[i]);
        }    
        
        WaitForMultipleObjects(capacity, readies, TRUE, INFINITE);
    }

    /////////////////////////////////////////
    void Recharge()
    {
    }

protected:

    /////////////////////////////////////////
    // should return false to stop thread execution
    virtual bool ProcessTask(int index) = 0;

    /////////////////////////////////////////
    // successor prototype:
    /*
    while(true)
    {
        prepare task
        if need, charge stop

        WaitResults();

        use results

        (if task was zero, then break)

        Recharge();
    }
    */
};

//-------------------------------------------------------------

DWORD GlobalThreadProcEGNUEPAFLL(LPVOID arg)
{
    ThreadArgsEGNUEPAFLL* pArg = (ThreadArgsEGNUEPAFLL*)arg;
    
        // create message queue and signal with evtReady
    
    MSG msg;
    PeekMessage(&msg, NULL, WM_USER, WM_USER, PM_NOREMOVE);

    SetEvent(pArg->pool->readies[pArg->index]);

    (pArg->pool)->ThreadProc(pArg->index);

    return 0;
}

//-------------------------------------------------------------
#endif // #ifdef _WIN32
#endif
