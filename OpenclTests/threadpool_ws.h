#ifndef THREADPOOLWS_H
#define THREADPOOLWS_H

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

union ThreadArgsEGNUEPAFLL
{    
    struct
    {
        ThreadPool* pool;
        int index;   
        SRWLOCK lock;
        bool taskPresent;
        bool resultsPresent; 
    }
    a;    

    char padding[64];
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

    bool* locked;

    /////////////////////////////////////////
    ThreadPool() : 
        thandles(NULL),
        tids(NULL),        
        readies(NULL),
        locked(NULL),
        array_run_args(NULL)
    {}

    /////////////////////////////////////////
    bool Init(size_t argCapacity)
    {
        capacity = argCapacity;

        masterThreadId = GetCurrentThreadId();

        thandles = (HANDLE*)malloc(capacity * sizeof(HANDLE));
        tids = (DWORD*)malloc(capacity * sizeof(DWORD));
        
        readies = (HANDLE*)malloc(capacity * sizeof(HANDLE));

        locked = (bool*)malloc(capacity * sizeof(bool));

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
            array_run_args[i].a.pool = this;
            array_run_args[i].a.index = i;  

            InitializeSRWLock(&(array_run_args[i].a.lock));
            
            array_run_args[i].a.taskPresent = false;
            array_run_args[i].a.resultsPresent = false;            
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
            CloseHandle(readies[i]);            
        }

        free(locked);
        free(thandles);    
        free(tids);
        free(readies);
        
        free(array_run_args);
    }
    
    /////////////////////////////////////////
    void ThreadProc(int index)
    {
        PSRWLOCK l = &(array_run_args[index].a.lock);

        bool* pTask = &(array_run_args[index].a.taskPresent); 
                
        while(true)
        {                    
            while(true)
            {
                bool task = false;

                if(TryAcquireSRWLockExclusive(l))
                {
                    task = *pTask;    
                    *pTask = false;
                    ReleaseSRWLockExclusive(l);
                }

                if(task)
                {
                    break;
                }
                
                YieldProcessor();
            };

            bool res = ProcessTask(index);

            //*

            while(true)
            {
                bool done = false;

                if(TryAcquireSRWLockExclusive(l))
                {
                    done = true;
                    array_run_args[index].a.resultsPresent = true;
                    ReleaseSRWLockExclusive(l);                    
                }

                if(done)
                {
                    break;
                }

                YieldProcessor();
            }

            //*/
            //SetEvent(readies[index]);
            
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
            locked[i] = false;
        }

        while(true)
        {
            for(size_t i = 0; i < capacity; ++i)
            {
                if(locked[i])
                {
                    continue;
                }

                PSRWLOCK l = &(array_run_args[i].a.lock);

                if(TryAcquireSRWLockExclusive(l))
                {
                    locked[i] = true;

                    array_run_args[i].a.taskPresent = true;    
                    array_run_args[i].a.resultsPresent = false;    

                    ReleaseSRWLockExclusive(l);
                }
            }

            size_t counter = 0;

            for(size_t i = 0; i < capacity; ++i)
            {
                counter += (locked[i] ? 1 : 0);
            }

            if(counter == capacity)
            {
                break;
            }      

            YieldProcessor();      
        }         
                    
        //WaitForMultipleObjects(capacity, readies, TRUE, INFINITE);

        //*      

        for(size_t i = 0; i < capacity; ++i)
        {
            locked[i] = false;
        }
        
        while(true)
        {
            for(size_t i = 0; i < capacity; ++i)
            {
                if(locked[i])
                {
                    continue;
                }
                
                PSRWLOCK l = &(array_run_args[i].a.lock);

                if(TryAcquireSRWLockExclusive(l))
                {
                    locked[i] = array_run_args[i].a.resultsPresent;

                    ReleaseSRWLockExclusive(l);
                }
            }

            size_t counter = 0;

            for(size_t i = 0; i < capacity; ++i)
            {
                counter += (locked[i] ? 1 : 0);
            }

            if(counter == capacity)
            {
                break;
            }            

            YieldProcessor();
        }         
        //*/
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
            
    SetEvent(pArg->a.pool->readies[pArg->a.index]);

    (pArg->a.pool)->ThreadProc(pArg->a.index);

    return 0;
}

//-------------------------------------------------------------
#endif // #ifdef _WIN32
#endif
