#ifndef THREADPOOLW_H
#define THREADPOOLW_H

#include <queue>
#include <utility>

//-------------------------------------------------------------

#ifdef _WIN32

#include <windows.h>

//-------------------------------------------------------------

size_t get_ncpu()
{
    SYSTEM_INFO si;

    GetSystemInfo(&si);

    return si.dwNumberOfProcessors;
}

inline void acquire_lock(bool* lock) 
{    
    while (InterlockedCompareExchange((volatile unsigned long*)lock, true, false));
}

inline void release_lock(bool* lock) 
{
    *lock = false;
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

    HANDLE* readies;

    ThreadArgsEGNUEPAFLL* array_run_args;

    DWORD masterThreadId;

    bool lock;

    bool stop;

    std::queue< std::pair<size_t, size_t> > items;

    /////////////////////////////////////////
    ThreadPool() : 
        thandles(NULL),
        tids(NULL),
        readies(NULL),
        array_run_args(NULL),
        lock(false),
        stop(false)        
    {}

    /////////////////////////////////////////
    bool Init(size_t argCapacity)
    {
        capacity = argCapacity;

        masterThreadId = GetCurrentThreadId();

        thandles = (HANDLE*)malloc(capacity * sizeof(HANDLE));
        tids = (DWORD*)malloc(capacity * sizeof(DWORD));

        readies = (HANDLE*)malloc(capacity * sizeof(HANDLE));

        array_run_args = (ThreadArgsEGNUEPAFLL*)malloc(capacity * sizeof(ThreadArgsEGNUEPAFLL));

        for(size_t i = 0; i < capacity; ++i)
        {
            readies[i] = CreateEvent(NULL, FALSE, FALSE, NULL);

            if(readies[i] == NULL)
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

            // create master thread message queue            
        MSG msg;
        PeekMessage(&msg, NULL, WM_USER, WM_USER, PM_NOREMOVE);

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
            CloseHandle(readies[i]);            
        }

        free(thandles);    
        free(tids);
        free(readies);    
        free(array_run_args);
    }
    
    /////////////////////////////////////////
    void ThreadProc(int index)
    {
        MSG msg;

        while(GetMessage(&msg, (HWND)-1, WM_APP, WM_APP) > 0)
        {
            if(stop)
            {
                PostThreadMessage(masterThreadId, WM_APP, 0, index);
                return;
            }                        
            
            do
            {
                acquire_lock(&lock);                  
                
                if(!items.empty())
                {
                    std::pair<size_t, size_t> item = items.front();                        
                    items.pop();                    
                    release_lock(&lock);
                    ProcessTask(/*index,*/ item);
                }
                else
                {
                    release_lock(&lock);
                    break;
                }                                                
            }
            while(true);               

            PostThreadMessage(masterThreadId, WM_APP, 0, index);
        }
    }

    /////////////////////////////////////////
    void WaitResults()
    {
            // charge task to all worker threads
            
        for(size_t i = 0; i < capacity; ++i)
        {
            PostThreadMessage(tids[i], WM_APP, 0, i);    
        }    
        
            // wait exactly 'capacity' messages from worker thread

        // to do: analyse if it is a safe strategy; if no, create an array to check indices

        MSG msg;

        size_t counter = 0;

        do
        {
            GetMessage(&msg, (HWND)-1, WM_APP, WM_APP);
            ++counter;
        }
        while(counter != capacity);        
    }

    /////////////////////////////////////////
    inline void Recharge()
    {
    }

protected:

    /////////////////////////////////////////
    virtual void ProcessTask(/*int index,*/ std::pair<size_t, size_t>& item) = 0;

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
