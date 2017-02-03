#ifndef THREADPOOLP_H
#define THREADPOOLP_H

//-------------------------------------------------------------
// aligned memory allocation cross-stuff
//-------------------------------------------------------------

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
// threads cross-stuff
//-------------------------------------------------------------

#ifdef _WIN32

#include <windows.h>

#define thread_sleep Sleep

typedef HANDLE thread_handle;

typedef DWORD thread_ret;
typedef void* thread_arg;

typedef HANDLE thread_mutex;

/*
typedef union
{    
    align_as(64) HANDLE mutex;
    char padding[64 - sizeof(HANDLE)];
}
thread_mutex;
*/

typedef thread_ret (*pthread_routine)(thread_arg);

size_t get_ncpu();

thread_handle create_thread(pthread_routine start_routine, thread_arg arg);
int thread_join(thread_handle, thread_ret* pStatus);

int thread_mutex_init();
int thread_mutex_destroy(thread_mutex* mut);
int thread_mutex_lock(thread_mutex* mut);
int thread_mutex_unlock(thread_mutex* mut);

void acquire_lock(bool* lock);
void release_lock(bool* lock);

void thread_yield_processor();

size_t get_ncpu()
{
    SYSTEM_INFO si;

    GetSystemInfo(&si);

    return si.dwNumberOfProcessors;
}

thread_handle create_thread(pthread_routine start_routine, thread_arg arg)
{
    DWORD tid;
    
    return CreateThread(NULL, 0, start_routine, arg, 0, &tid);
}

int thread_join(thread_handle thread, thread_ret* pStatus)
{    
    *pStatus = 0;
    return WaitForSingleObject(thread, INFINITE);
}

int thread_mutex_init(thread_mutex* mut)
{
    return (*mut = CreateMutex(NULL, FALSE, NULL)) ? 0 : 1;
    //return (mut->mutex = CreateMutex(NULL, FALSE, NULL)) ? 0 : 1;
}

int thread_mutex_destroy(thread_mutex* mut)
{
    return CloseHandle(*mut) ? 0 : 1;
    //return CloseHandle(mut->mutex) ? 0 : 1;
}

inline int thread_mutex_lock(thread_mutex* mut)
{
    return WaitForSingleObject(*mut, INFINITE);
    //return WaitForSingleObject(mut->mutex, INFINITE);
}

inline int thread_mutex_unlock(thread_mutex* mut)
{
    return ReleaseMutex(*mut) ? 0 : 1;
    //return ReleaseMutex(mut->mutex) ? 0 : 1;
}

//inline void acquire_lock(bool* lock) 
inline void acquire_lock(PSRWLOCK lock) 
{    
    /*
    while(InterlockedCompareExchange((volatile LONG*)lock, true, false))
    {
        _mm_pause();
    }
    */
    AcquireSRWLockExclusive(lock);    
}

//inline void release_lock(bool* lock) 
inline void release_lock(PSRWLOCK lock) 
{
    //*lock = false;
    ReleaseSRWLockExclusive(lock);
}

inline void thread_yield_processor()
{
    _mm_pause();
}

#else // POSIX stuff

#include <time.h>
#include <sys/types.h>
#include <sys/sysctl.h>
#include <pthread.h>
#include <sched.h>

void thread_sleep(size_t ms);

void thread_sleep(size_t ms)
{
    size_t seconds = ms / 1000;

    struct timespec tim, tim2;

    tim.tv_sec = seconds;
    tim.tv_nsec = (ms - seconds * 1000) * 1000000;    

    nanosleep(&tim , &tim2);
}

typedef pthread_t thread_handle;

typedef void* thread_ret;
typedef void* thread_arg;

typedef thread_ret (*pthread_routine)(thread_arg);

typedef pthread_mutex_t thread_mutex;

size_t get_ncpu();

thread_handle create_thread(pthread_routine start_routine, thread_arg arg);
int thread_join(thread_handle, thread_ret* pStatus);
int thread_mutex_init();

void acquire_lock(bool* lock);
void release_lock(bool* lock);

//#define thread_yield_processor pthread_yield;
#define thread_yield_processor sched_yield

#define thread_mutex_destroy pthread_mutex_destroy
#define thread_mutex_lock pthread_mutex_lock
#define thread_mutex_unlock pthread_mutex_unlock

size_t get_ncpu()
{
    size_t ncpu = 0;
    size_t len = sizeof(int);

    sysctlbyname("hw.ncpu", &ncpu,	&len, NULL, 0);

    return ncpu;
}

thread_handle create_thread(pthread_routine start_routine, thread_arg arg)
{
    thread_handle th;

    pthread_create(&th, NULL, start_routine, arg);

    return th;
}

int thread_join(thread_handle thread, thread_ret* pStatus)
{
    return pthread_join(thread, pStatus);
}

int thread_mutex_init(thread_mutex* mut)
{
    return pthread_mutex_init(mut, NULL);
}

inline void acquire_lock(bool* lock) 
{
    while (!__sync_bool_compare_and_swap(lock, false, true))
    {
        asm("PAUSE"); 
    }
}

inline void release_lock(bool* lock) 
{
    *lock = false;
}

#endif

//-------------------------------------------------------------

#include <queue>
#include <utility>

//-------------------------------------------------------------

struct ThreadPool;

struct ThreadArgsQDAIEOPJKL
{
    ThreadPool* pool;
    int index;
    bool ready;
};

thread_ret GlobalThreadProcQDAIEOPJKL(thread_arg arg);

//-------------------------------------------------------------

struct ThreadPool
{
    size_t capacity;

    thread_handle* thandles;

    thread_mutex* array_e1;
    thread_mutex* array_e1s;
    thread_mutex* array_e2;

    ThreadArgsQDAIEOPJKL* array_run_args;

#ifdef _WIN32

    SRWLOCK lock;
    //align_as(16) bool lock;

#else

    bool lock;

#endif 

    bool stop;
    
    std::queue< std::pair<size_t, size_t> > items;

    /////////////////////////////////////////
    ThreadPool() : 
        thandles(NULL),
        array_e1(NULL),
        array_e1s(NULL),
        array_e2(NULL),
        array_run_args(NULL),        
        stop(false)
    {}

    /////////////////////////////////////////
    bool Init(size_t argCapacity)
    {
        capacity = argCapacity;

        thandles = (thread_handle*)malloc(capacity * sizeof(thread_handle));

        array_e1 = (thread_mutex*)malloc(capacity * sizeof(thread_mutex));
        array_e1s = (thread_mutex*)malloc(capacity * sizeof(thread_mutex));
        array_e2 = (thread_mutex*)malloc(capacity * sizeof(thread_mutex));

        array_run_args = (ThreadArgsQDAIEOPJKL*)malloc(capacity * sizeof(ThreadArgsQDAIEOPJKL));

            // init all mutexes
        for(size_t i = 0; i < capacity; ++i)
        {
            if(thread_mutex_init(array_e1 + i))
            {
                return false;
            }

            if(thread_mutex_init(array_e1s + i))
            {
                return false;
            }

            if(thread_mutex_init(array_e2 + i))
            {
                return false;
            }
        }

            // lock all newly created mutexes
        for(size_t i = 0; i < capacity; ++i)
        {
            thread_mutex_lock(array_e1 + i);
            thread_mutex_lock(array_e1s + i);
            thread_mutex_lock(array_e2 + i);
        }
        
            // all newly created threads block on mutexes
        for(size_t i = 0; i < capacity; ++i)
        {
            array_run_args[i].pool = this;
            array_run_args[i].index = i;
            array_run_args[i].ready = false;

            create_thread(GlobalThreadProcQDAIEOPJKL, (thread_arg)&(array_run_args[i]));
        }
            // ensure threads are created and blocked on e1, e1s to this moment
        for(size_t i = 0; i < capacity; ++i)
        {
            while(!array_run_args[i].ready)
            {                                    
                thread_yield_processor();            
            }
        }
        
        thread_yield_processor();            

            // ... yet, we can't be sure for 100%; OK.

            // unlock mutexes, let blocked threads run
        for(size_t i = 0; i < capacity; ++i)    
        {            
            thread_mutex_unlock(array_e1 + i);
            thread_mutex_unlock(array_e1s + i);
        }

#ifdef _WIN32
        InitializeSRWLock(&lock);
        //lock = false;
#else
        lock = false;        
#endif
        return true;
    }

    /////////////////////////////////////////
    void Dispose()
    {
        thread_ret r;

        for(size_t i = 0; i < capacity; ++i)
        {
            thread_join(thandles[i], &r);
        }

        for(size_t i = 0; i < capacity; ++i)
        {
            thread_mutex_destroy(array_e1 + i);
            thread_mutex_destroy(array_e1s + i);
            thread_mutex_destroy(array_e2 + i);
        }

        free(thandles);    

        free(array_e1);
        free(array_e1s);
        free(array_e2);    

        free(array_run_args);
    }
    
    /////////////////////////////////////////
    void ThreadProc(int index)
    {
        thread_mutex* pe1 = array_e1 + index;
        thread_mutex* pe1s = array_e1s + index;
        thread_mutex* pe2 = array_e2 + index;
        
        thread_mutex_lock(pe1);
        thread_mutex_lock(pe1s);

        do
        {
            thread_mutex_unlock(pe1);
            thread_mutex_lock(pe2);

            if(stop)
            {
                thread_mutex_unlock(pe1s);    
                thread_mutex_unlock(pe2);

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

            thread_mutex_unlock(pe1s);
            thread_mutex_lock(pe1);
            thread_mutex_unlock(pe2);
            thread_mutex_lock(pe1s);            
        }
        while(true);
    }

    /////////////////////////////////////////
    void WaitResults()
    {
        size_t i;

        for(i = 0; i < capacity; ++i)
        {
            thread_mutex_lock(array_e1 + i);    
        }    

        for(i = 0; i < capacity; ++i)
        {
            thread_mutex_unlock(array_e2 + i);    
        }            

        for(i = 0; i < capacity; ++i)
        {
            thread_mutex_lock(array_e1s + i);    
        }                    
    }

    /////////////////////////////////////////
    void Recharge()
    {
        size_t i;

        for(i = 0; i < capacity; ++i)
        {
            thread_mutex_unlock(array_e1 + i);    
        }    

        for(i = 0; i < capacity; ++i)
        {
            thread_mutex_lock(array_e2 + i);    
        }            

        for(i = 0; i < capacity; ++i)
        {
            thread_mutex_unlock(array_e1s + i);    
        }                    
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
        push N pairs to queue        

        WaitResults();

        use results

        (if task was zero, then break)

        Recharge();
    }
    */
};

//-------------------------------------------------------------

thread_ret GlobalThreadProcQDAIEOPJKL(thread_arg arg)
{
    ThreadArgsQDAIEOPJKL* pArg = (ThreadArgsQDAIEOPJKL*)arg;

    pArg->ready = true;

    (pArg->pool)->ThreadProc(pArg->index);

    return NULL;
}

//-------------------------------------------------------------

#endif
