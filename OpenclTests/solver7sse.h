#ifndef SOLVERSSE_H
#define SOLVERSSE_H

//-------------------------------------------------------------
// aligned memory allocation cross-stuff

#include <stdlib.h>

#ifdef _WIN32

#include <malloc.h>

#define aligned_alloc(alignment, size) _aligned_malloc((size), (alignment))
#define aligned_free _aligned_free
#define align_as(alignment) __declspec(align((alignment)))

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

#ifdef _WIN32

//-------------------------------------------------------------
// threads cross-stuff

#include <windows.h>

#else

#include <sys/types.h>
#include <sys/sysctl.h>
#include <pthread.h>

typedef pthread_t thread_handle;

size_t get_ncpu();

size_t get_ncpu()
{
    size_t ncpu = 0;
    size_t len = sizeof(int);

    sysctlbyname("hw.ncpu", &ncpu,	&len, NULL, 0);

    return ncpu;
}

thread_handle create_thread(void* (*start_routine)(void*), void* arg);

thread_handle create_thread(void* (*start_routine)(void*), void* arg)
{
    thread_handle th;

    pthread_create(&th, NULL, start_routine, NULL);

    return th;
}

int join_thread(thread_handle, void** pStatus);

int join_thread(thread_handle thread, void** pStatus)
{
    return pthread_join(thread, pStatus);
}

#endif

//-------------------------------------------------------------

void* thread_proc(void* arg)
{
    return arg;
}

//-------------------------------------------------------------

#include <xmmintrin.h>

//-------------------------------------------------------------

struct Solver
{
    size_t sseAlignment;
    size_t sseBaseCount;
    size_t sseBlockStride;

    size_t ncpu;

    size_t dimension;

    size_t expandedDimension;

    size_t extra;

    size_t sseBlocksCount;
    
    size_t fp32VectorSize;    
    size_t fp64VectorSize;    

    size_t fp32VectorStride;
    size_t fp32ResourceStride;

    size_t fp64VectorStride;
    size_t fp64ResourceStride;

    float* fp32Matrix;    
    double* fp64Matrix;

    double* fp64Vector;

    /////////////////////////////////////////
    Solver() :
        sseAlignment(16),
        sseBaseCount(4),
        sseBlockStride(16),
        fp32Matrix(NULL),
        fp64Matrix(NULL),
        fp64Vector(NULL)
    {}

    /////////////////////////////////////////
    void Dispose()
    {        
        aligned_free(fp32Matrix);     
        aligned_free(fp64Matrix);   
        aligned_free(fp64Vector);     
    }

    /////////////////////////////////////////
    bool Init(size_t useDimension)
    {
        dimension = useDimension;

        fp32VectorSize = dimension * sizeof(float);        
        fp64VectorSize = dimension * sizeof(double);

        extra = dimension % sseBaseCount;

        expandedDimension = (extra == 0) ? dimension : dimension + (sseBaseCount - extra);

        sseBlocksCount = expandedDimension / sseBaseCount;

        fp32VectorStride = expandedDimension * sizeof(float);        
        fp32ResourceStride = dimension * fp32VectorStride;

        fp64VectorStride = expandedDimension * sizeof(double);
        fp64ResourceStride = dimension * fp64VectorStride;

        fp32Matrix = (float*)aligned_alloc(sseAlignment, fp32ResourceStride); 
        fp64Matrix = (double*)aligned_alloc(sseAlignment, fp64ResourceStride); 

        fp64Vector = (double*)aligned_alloc(sseAlignment, fp64VectorStride); 

        ncpu = get_ncpu();

        return true;
    }

    /////////////////////////////////////////
    void useMatrix(float* argMatrix)
    {
            // copy initial matrix transposed
        
        float* src;
        float* dest = fp32Matrix;

        size_t trail = expandedDimension - dimension;

        size_t stop = (trail == 0) ? sseBlocksCount : (sseBlocksCount - 1);        
        
        align_as(16) float buf[4]; 
        __m128* pBuf = (__m128*)buf;

            // scan cols of initial matrix, store rows of transposed matrix
        for(size_t col = 0; col < dimension; ++col)
        {
            src = argMatrix + col;

            for(size_t blockIndex = 0; blockIndex < stop; ++blockIndex)
            {
                    // accumulate elements in temporary array
                for(size_t i = 0; i < sseBaseCount; ++i, src += dimension)
                {
                    buf[i] = *src;                    
                }
                    // then store whole block
                _mm_store_ps(dest, *pBuf);                
                dest += sseBaseCount;
            }
        
            if(extra)
            {
                    // accumulate last block in column of initial matrix 
                    //  (block will not be full)
                for(size_t i = 0; i < extra; ++i, src += dimension)
                {
                    buf[i] = *src;                    
                }

                    // then store whole block
                _mm_store_ps(dest, *pBuf);                
                dest += sseBaseCount;
            }

        } // end for col     

            // expand fp32 transposed matrix to fp64 
            // (also transposed, by relation to initial matrix)
        for(size_t row = 0, index = 0; row < dimension; ++row)
        {
            for(size_t col = 0; col < dimension; ++col, ++index)
            {                
                fp64Matrix[index] = (double)(fp32Matrix[index]);    
            }

                // skip copying trailing bytes
            index += trail;
        }        
    }

    /////////////////////////////////////////
    void useVector(float* argVector)
    {
        for(size_t i = 0; i < dimension; ++i)
        {
            fp64Vector[i] = (double)(argVector[i]);
        }
    }

    /////////////////////////////////////////
    bool Solve()
    {
        /*
        thread_handle th = create_thread(thread_proc, NULL);
        void* result;
        join_thread(th, &result);
        */

        for(size_t step = 0; step < dimension; ++step)
        {            
            int pivotIndex = findPivot(step);

            if(pivotIndex < 0)
            {
                return false;
            }
            
            if(pivotIndex != step)
            {
                // swap cols
            }
            
            // divide elements in 'step' row
            // process main block of values 
        }

        return true;
    }

private:

    /////////////////////////////////////////
    int findPivot(size_t step)
    {
        int pivotIndex = -1;        
        float maxValue = 0;

        for(size_t col = step, scanIndex = step * expandedDimension + col; col < dimension; ++col, ++scanIndex)
        {
            float fv = fp32Matrix[scanIndex];
            float fav = fv;

                // clear sign bit to get absolute value
            ((char*)&fav)[3] &= '\x7F';                
            
            if(fav > maxValue)
            {
                maxValue = fav;
                pivotIndex = col;
            }
        }

        return pivotIndex;
    }        
};

//-------------------------------------------------------------

#endif
